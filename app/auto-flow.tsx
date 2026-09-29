"use client";

import { decompressFrames, parseGIF } from "gifuct-js";
import { useRef, useState } from "react";

type Asset = { index: number; mime: string; ext: string; bytes: string };
type Crop = { dataUrl: string; width: number; height: number; bytes: number };
type Worker = { recognize: (image: File) => Promise<{ data: { text: string; confidence: number } }>; terminate: () => Promise<void> };
type FaceCandidate = ReturnType<typeof analyze> & { image: HTMLImageElement; ext: string; asset: Asset; frameIndex: number };
type CaseResult = { code: string; name: string; crops: Crop[] };
type BatchFailure = { code: string; name: string; message: string };
type SaveDirectory = { name: string; queryPermission?: (options?: { mode?: "read" | "readwrite" }) => Promise<PermissionState>; requestPermission?: (options?: { mode?: "read" | "readwrite" }) => Promise<PermissionState>; getDirectoryHandle: (name: string, options?: { create?: boolean }) => Promise<SaveDirectory>; getFileHandle: (name: string, options?: { create?: boolean }) => Promise<{ createWritable: () => Promise<{ write: (data: Uint8Array) => Promise<void>; close: () => Promise<void> }> }>; };

declare global {
  interface Window {
    Tesseract?: { createWorker: (lang: string, oem: number, options: Record<string, unknown>) => Promise<Worker> };
    JSZip?: new () => { folder: (name: string) => { file: (name: string, data: string, options: { base64: boolean }) => void } | null; generateAsync: (options: { type: "blob"; compression: "STORE" }) => Promise<Blob> };
  }
}

function parseReport(text: string) {
  const lines = text.normalize("NFKC").split(/\r?\n/).map(line => line.replace(/[\s\u3000]/g, ""));
  const compact = lines.join("");
  const numberLine = lines.find(line => line.includes("超声号")) ?? compact;
  const numberSource = numberLine.includes("超声号") ? numberLine.slice(numberLine.indexOf("超声号") + 3) : numberLine;
  const code = numberSource.match(/20\d{8,10}/)?.[0] ?? null;
  const nameLine = lines.find(line => line.includes("姓名")) ?? compact;
  let name: string | null = null;
  if (nameLine.includes("姓名")) {
    let source = nameLine.slice(nameLine.indexOf("姓名") + 2).replace(/^[：:=]/, "");
    source = source.split(/性别|年龄|门诊号|超声号|申请科室|检查部位|申请医生|孕龄/)[0];
    name = source.match(/[\u3400-\u9fff·]{2,5}/)?.[0] ?? null;
  }
  return { code, name };
}

const clamp = (n: number, min: number, max: number) => Math.max(min, Math.min(max, n));
// Keep a little more surrounding image than the old close-up crop, while
// using this same face-to-frame ratio for every exported album image.
// Keep a useful amount of surrounding ultrasound context. A smaller ratio
// means the face occupies less of the final frame, which prevents the
// "big-head" crop and gives the crop planner room to keep the whole face.
const ALBUM_FACE_RATIO = .56;
const BATCH_QUERY_GAP_MS = 1600;

function loadImage(src: string): Promise<HTMLImageElement> {
  return new Promise((resolve, reject) => {
    const image = new Image(); image.onload = () => resolve(image); image.onerror = reject; image.src = src;
  });
}

function decodeBytes(base64: string) {
  const binary = atob(base64);
  const bytes = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i);
  return bytes.buffer;
}

async function decodeGifFrames(asset: Asset): Promise<Array<{ image: HTMLImageElement; frameIndex: number }>> {
  const parsed = parseGIF(decodeBytes(asset.bytes));
  const frames = decompressFrames(parsed, true);
  const width = parsed.lsd.width, height = parsed.lsd.height;
  if (!width || !height || !frames.length) throw new Error("GIF 没有可读取的影像帧。");

  const canvas = document.createElement("canvas"); canvas.width = width; canvas.height = height;
  const ctx = canvas.getContext("2d"); if (!ctx) throw new Error("无法合成 GIF 影像帧。");
  ctx.fillStyle = "#000";
  ctx.fillRect(0, 0, width, height);
  const images: Array<{ image: HTMLImageElement; frameIndex: number }> = [];
  // Match the manual Convertio workflow: evaluate every GIF frame, not a
  // sparse sample, so a short clear moment is never skipped.
  const step = 1;

  for (let index = 0; index < frames.length; index++) {
    const frame = frames[index] as {
      dims: { left: number; top: number; width: number; height: number };
      patch: Uint8ClampedArray;
      disposalType?: number;
    };
    const before = frame.disposalType === 3 ? ctx.getImageData(0, 0, width, height) : null;
    const patch = ctx.createImageData(frame.dims.width, frame.dims.height);
    patch.data.set(frame.patch);
    // GIF frames usually contain only a small rectangle. Draw the patch through
    // an offscreen canvas so transparent pixels preserve the previous frame
    // instead of erasing it and creating white/fragmented holes.
    const patchCanvas = document.createElement("canvas");
    patchCanvas.width = frame.dims.width; patchCanvas.height = frame.dims.height;
    const patchCtx = patchCanvas.getContext("2d");
    if (!patchCtx) throw new Error("无法读取 GIF 局部帧。");
    patchCtx.putImageData(patch, 0, 0);
    ctx.drawImage(patchCanvas, frame.dims.left, frame.dims.top);
    if (index % step === 0 || index === frames.length - 1) {
      // Flatten each composited frame to a standalone JPG, just like the
      // manual conversion workflow. This also removes transparent GIF pixels
      // before the normal still-image face analysis runs.
      images.push({ image: await loadImage(canvas.toDataURL("image/jpeg", .96)), frameIndex: index });
    }

    if (frame.disposalType === 2) {
      ctx.fillStyle = "#000";
      ctx.fillRect(frame.dims.left, frame.dims.top, frame.dims.width, frame.dims.height);
    }
    else if (frame.disposalType === 3 && before) ctx.putImageData(before, 0, 0);
  }
  return images;
}

function analyze(image: HTMLImageElement) {
  const width = image.naturalWidth, height = image.naturalHeight, sw = 160, sh = Math.max(1, Math.round(160 * height / width));
  const canvas = document.createElement("canvas"); canvas.width = sw; canvas.height = sh;
  const ctx = canvas.getContext("2d", { willReadFrequently: true });
  if (!ctx) throw new Error("无法分析影像。");
  ctx.drawImage(image, 0, 0, sw, sh);
  const pixels = ctx.getImageData(0, 0, sw, sh).data, mask = new Uint8Array(sw * sh), seen = new Uint8Array(sw * sh), queue = new Int32Array(sw * sh);
  let edgeEnergy = 0, edgeSamples = 0;
  const grayAt = (x: number, y: number) => {
    const i = (y * sw + x) * 4;
    return pixels[i] * .299 + pixels[i + 1] * .587 + pixels[i + 2] * .114;
  };
  for (let y = 1; y < sh - 1; y++) for (let x = 1; x < sw - 1; x++) {
    const center = grayAt(x, y);
    const laplacian = 4 * center - grayAt(x - 1, y) - grayAt(x + 1, y) - grayAt(x, y - 1) - grayAt(x, y + 1);
    edgeEnergy += laplacian * laplacian; edgeSamples++;
  }
  const edgeRms = edgeSamples ? Math.sqrt(edgeEnergy / edgeSamples) : 0;
  const clarity = clamp((edgeRms - 5) / 34, 0, 1);
  const left = Math.floor(sw * .08), right = Math.ceil(sw * .99), top = Math.floor(sh * .12), bottom = Math.ceil(sh * .98);
  let warmCount = 0;
  for (let y = top; y < bottom; y++) for (let x = left; x < right; x++) {
    const i = (y * sw + x) * 4, r = pixels[i], g = pixels[i + 1], b = pixels[i + 2];
    if (r > 48 && g > 28 && b > 10 && r > g * 1.06 && g > b * 1.02 && r - b > 18) { mask[y * sw + x] = 1; warmCount++; }
  }
  const empty = { score: 0, frontal: 0, cx: width * .5, cy: height * .42, faceWidth: 0, faceHeight: 0, faceLeft: 0, faceRight: 0, faceTop: 0, faceBottom: 0, colorPixels: warmCount, clarity, featureScore: 0, faceConfidence: 0 };
  if (warmCount < 80) return empty;
  let largest: { count: number; points: Int32Array; minX: number; maxX: number; minY: number; maxY: number } | null = null;
  for (let seed = 0; seed < mask.length; seed++) {
    if (!mask[seed] || seen[seed]) continue;
    let head = 0, tail = 0, minX = sw, maxX = 0, minY = sh, maxY = 0;
    queue[tail++] = seed; seen[seed] = 1;
    while (head < tail) {
      const p = queue[head++], x = p % sw, y = Math.floor(p / sw);
      minX = Math.min(minX, x); maxX = Math.max(maxX, x); minY = Math.min(minY, y); maxY = Math.max(maxY, y);
      const a = x > left ? p - 1 : -1, b = x + 1 < right ? p + 1 : -1, c = y > top ? p - sw : -1, d = y + 1 < bottom ? p + sw : -1;
      for (const next of [a, b, c, d]) if (next >= 0 && mask[next] && !seen[next]) { seen[next] = 1; queue[tail++] = next; }
    }
    if (!largest || tail > largest.count) largest = { count: tail, points: queue.slice(0, tail), minX, maxX, minY, maxY };
  }
  if (!largest || largest.count < 80) return empty;
  const largestMask = new Uint8Array(sw * sh); for (const p of largest.points) largestMask[p] = 1;
  // The jaw/ear line is usually below the eye-and-nose area. The previous
  // 57% cutoff stopped at the cheeks on several frames, so keep enough of the
  // upper component to include the whole lower face without taking its feet
  // or unrelated bottom structures as the face.
  const faceBottom = Math.min(largest.maxY, Math.round(largest.minY + (largest.maxY - largest.minY) * .74));
  let pixelsInFace = 0, faceMinX = sw, faceMaxX = 0;
  for (const p of largest.points) {
    const x = p % sw, y = Math.floor(p / sw); if (y > faceBottom) continue;
    pixelsInFace++; faceMinX = Math.min(faceMinX, x); faceMaxX = Math.max(faceMaxX, x);
  }
  if (pixelsInFace < 80 || faceMaxX - faceMinX < 18) return empty;
  // The largest warm component can contain an arm or placenta beside the
  // face. Do not choose the densest part: arms and placentas are often the
  // densest part. Score each horizontal window for the paired dark details
  // and central lower detail that a frontal face normally contains.
  const rawFaceWidth = faceMaxX - faceMinX + 1;
  const windowWidth = Math.max(18, Math.round(rawFaceWidth * .62));
  let bestFaceStart = faceMinX, bestFaceQuality = -Infinity;
  for (let start = faceMinX; start + windowWidth - 1 <= faceMaxX; start++) {
    const end = start + windowWidth - 1, center = (start + end) / 2, faceHeight = faceBottom - largest.minY + 1;
    let leftCount = 0, rightCount = 0, both = 0, rowCount = 0, rows = 0;
    for (let y = largest.minY; y <= faceBottom; y++) {
      let rowBoth = 0;
      for (let dx = 2; dx <= windowWidth / 2; dx++) {
        const lx = Math.round(center - dx), rx = Math.round(center + dx);
        if (lx < start || rx > end) continue;
        const a = largestMask[y * sw + lx], b = largestMask[y * sw + rx];
        leftCount += a; rightCount += b; if (a && b) { both++; rowBoth++; }
      }
      rows++; if (rowBoth) rowCount++;
    }
    const frontal = leftCount + rightCount ? 2 * both / (leftCount + rightCount) : 0;
    const balance = leftCount + rightCount ? 2 * Math.min(leftCount, rightCount) / (leftCount + rightCount) : 0;
    const coverage = rows ? rowCount / rows : 0;
    let darkLeft = 0, darkRight = 0, eyeLeft = 0, eyeRight = 0, mouthCenter = 0;
    const featureTop = Math.round(largest.minY + faceHeight * .08), featureBottom = Math.round(largest.minY + faceHeight * .68);
    const eyeTop = Math.round(largest.minY + faceHeight * .18), eyeBottom = Math.round(largest.minY + faceHeight * .52);
    const mouthTop = Math.round(largest.minY + faceHeight * .52), mouthBottom = Math.round(largest.minY + faceHeight * .82);
    const mouthLeft = Math.round(start + windowWidth * .25), mouthRight = Math.round(end - windowWidth * .25);
    for (let y = featureTop; y <= featureBottom; y++) for (let x = start; x <= end; x++) {
      const i = (y * sw + x) * 4, luminance = pixels[i] * .299 + pixels[i + 1] * .587 + pixels[i + 2] * .114;
      if (luminance >= 62) continue;
      let warmNeighbors = 0;
      for (let ny = Math.max(0, y - 2); ny <= Math.min(sh - 1, y + 2); ny++) for (let nx = Math.max(0, x - 2); nx <= Math.min(sw - 1, x + 2); nx++) warmNeighbors += largestMask[ny * sw + nx];
      if (warmNeighbors < 4) continue;
      if (x < center) darkLeft++; else darkRight++;
      if (y >= eyeTop && y <= eyeBottom) { if (x < center) eyeLeft++; else eyeRight++; }
      if (y >= mouthTop && y <= mouthBottom && x >= mouthLeft && x <= mouthRight) mouthCenter++;
    }
    const darkTotal = darkLeft + darkRight, eyeTotal = eyeLeft + eyeRight;
    const darkBalance = darkTotal ? 2 * Math.min(darkLeft, darkRight) / darkTotal : 0;
    const darkPresence = Math.min(1, darkTotal / 32);
    const eyePair = eyeTotal ? 2 * Math.min(eyeLeft, eyeRight) / eyeTotal : 0;
    const eyePresence = Math.min(1, eyeTotal / 28);
    const mouthPresence = Math.min(1, mouthCenter / 18);
    const featureScore = clamp(darkBalance * darkPresence * .35 + eyePair * eyePresence * .45 + mouthPresence * .20, 0, 1);
    const quality = featureScore * .56 + frontal * .20 + balance * .10 + coverage * .06 + (windowWidth / rawFaceWidth) * .08;
    if (quality > bestFaceQuality) { bestFaceQuality = quality; bestFaceStart = start; }
  }
  faceMinX = bestFaceStart;
  faceMaxX = bestFaceStart + windowWidth - 1;
  let localPixels = 0;
  for (const p of largest.points) {
    const x = p % sw, y = Math.floor(p / sw);
    if (y <= faceBottom && x >= faceMinX && x <= faceMaxX) localPixels++;
  }
  if (localPixels < 40) return empty;
  const faceWidth = faceMaxX - faceMinX + 1, faceHeight = faceBottom - largest.minY + 1;
  // Anchor the crop to the selected facial window, not to the warm-pixel
  // centroid, because the centroid is pulled toward a touching arm/placenta.
  const cx = (faceMinX + faceMaxX) / 2, cy = largest.minY + faceHeight * .46;
  let leftCount = 0, rightCount = 0, both = 0, rowCount = 0, rows = 0;
  for (let y = largest.minY; y <= faceBottom; y++) {
    let rowBoth = 0;
    for (let dx = 2; dx <= faceWidth / 2; dx++) {
      const lx = Math.round(cx - dx), rx = Math.round(cx + dx); if (lx < left || rx >= right) continue;
      const a = largestMask[y * sw + lx], b = largestMask[y * sw + rx]; leftCount += a; rightCount += b; if (a && b) { both++; rowBoth++; }
    }
    rows++; if (rowBoth) rowCount++;
  }
  const frontal = leftCount + rightCount ? 2 * both / (leftCount + rightCount) : 0;
  const balance = leftCount + rightCount ? 2 * Math.min(leftCount, rightCount) / (leftCount + rightCount) : 0;
  const coverage = rows ? rowCount / rows : 0;
  let darkLeft = 0, darkRight = 0, eyeLeft = 0, eyeRight = 0, mouthCenter = 0;
  const featureTop = Math.round(largest.minY + faceHeight * .08), featureBottom = Math.round(largest.minY + faceHeight * .68);
  const featureLeft = Math.round(faceMinX + faceWidth * .14), featureRight = Math.round(faceMaxX - faceWidth * .14);
  const eyeTop = Math.round(largest.minY + faceHeight * .18), eyeBottom = Math.round(largest.minY + faceHeight * .52);
  const mouthTop = Math.round(largest.minY + faceHeight * .52), mouthBottom = Math.round(largest.minY + faceHeight * .82);
  const mouthLeft = Math.round(faceMinX + faceWidth * .25), mouthRight = Math.round(faceMaxX - faceWidth * .25);
  for (let y = featureTop; y <= featureBottom; y++) for (let x = featureLeft; x <= featureRight; x++) {
    const i = (y * sw + x) * 4, luminance = pixels[i] * .299 + pixels[i + 1] * .587 + pixels[i + 2] * .114;
    if (luminance >= 62) continue;
    let warmNeighbors = 0;
    for (let ny = Math.max(0, y - 2); ny <= Math.min(sh - 1, y + 2); ny++) for (let nx = Math.max(0, x - 2); nx <= Math.min(sw - 1, x + 2); nx++) warmNeighbors += largestMask[ny * sw + nx];
    if (warmNeighbors < 4) continue;
    if (x < cx) darkLeft++; else darkRight++;
    if (y >= eyeTop && y <= eyeBottom) { if (x < cx) eyeLeft++; else eyeRight++; }
    if (y >= mouthTop && y <= mouthBottom && x >= mouthLeft && x <= mouthRight) mouthCenter++;
  }
  const darkTotal = darkLeft + darkRight;
  const darkBalance = darkTotal ? 2 * Math.min(darkLeft, darkRight) / darkTotal : 0;
  const darkPresence = Math.min(1, darkTotal / 32);
  const eyeTotal = eyeLeft + eyeRight;
  const eyePair = eyeTotal ? 2 * Math.min(eyeLeft, eyeRight) / eyeTotal : 0;
  const eyePresence = Math.min(1, eyeTotal / 28);
  const mouthPresence = Math.min(1, mouthCenter / 18);
  // A frontal face should show a balanced pair of eye-area details and a
  // central lower facial detail, not just a large warm-colored blob.
  const featureScore = clamp(darkBalance * darkPresence * .35 + eyePair * eyePresence * .45 + mouthPresence * .20, 0, 1);
  const faceConfidence = frontal * .36 + coverage * .10 + balance * .08 + clarity * .14 + featureScore * .32;
  return {
    score: frontal * .42 + coverage * .14 + balance * .08 + clarity * .10 + featureScore * .26,
    frontal,
    cx: cx / sw * width,
    cy: cy / sh * height,
    faceWidth,
    faceHeight,
    // Retain the detected face box in source-image pixels. The crop planner
    // uses this box to prove that the face fits before exporting anything.
    faceLeft: faceMinX / sw * width,
    faceRight: (faceMaxX + 1) / sw * width,
    faceTop: largest.minY / sh * height,
    faceBottom: (faceBottom + 1) / sh * height,
    colorPixels: warmCount,
    clarity,
    featureScore,
    faceConfidence,
  };
}

async function snapshot(asset: Asset): Promise<FaceCandidate[]> {
  if (/^\.?gif$/i.test(asset.ext)) {
    const images = await decodeGifFrames(asset);
    return images.map(({ image, frameIndex }) => {
      const result = analyze(image);
      return { ...result, image, score: result.score * .82 + Math.min(1, Math.sqrt(image.naturalWidth * image.naturalHeight / (1024 * 768))) * .18, ext: asset.ext, asset, frameIndex };
    });
  }
  const image = await loadImage(`data:${asset.mime};base64,${asset.bytes}`);
  const result = analyze(image);
  return [{ ...result, image, score: result.score * .82 + Math.min(1, Math.sqrt(image.naturalWidth * image.naturalHeight / (1024 * 768))) * .18, ext: asset.ext, asset, frameIndex: 0 }];
}

function hasCompleteCropRoom(candidate: FaceCandidate) {
  const width = candidate.image.naturalWidth, height = candidate.image.naturalHeight;
  const faceWidthPixels = candidate.faceWidth * width / 160;
  const maxCropWidth = maxCropWidthFor(candidate);
  const cropWidth = Math.min(Math.floor(faceWidthPixels / ALBUM_FACE_RATIO / 3) * 3, maxCropWidth);
  const cropHeight = cropWidth * 4 / 3;
  const boxWidth = Math.max(1, candidate.faceRight - candidate.faceLeft);
  const boxHeight = Math.max(1, candidate.faceBottom - candidate.faceTop);
  const expandedLeft = candidate.faceLeft - boxWidth * .12;
  const expandedRight = candidate.faceRight + boxWidth * .12;
  const expandedTop = candidate.faceTop - boxHeight * .12;
  const expandedBottom = candidate.faceBottom + boxHeight * .12;
  return cropWidth >= 120 && cropHeight >= boxHeight && expandedRight - expandedLeft <= cropWidth && expandedBottom - expandedTop <= cropHeight && expandedLeft >= 0 && expandedRight <= width && expandedTop >= 0 && expandedBottom <= height;
}

function chooseFaceCandidates(candidates: FaceCandidate[]) {
  const structural = candidates.filter(candidate => {
    const width = candidate.image.naturalWidth, height = candidate.image.naturalHeight;
    const faceWidthPixels = candidate.faceWidth * width / 160;
    const maxCropWidth = Math.floor(Math.min(width, (Math.floor(height * .96) - Math.ceil(height * .14)) * .75, Math.ceil(width * .98) - Math.floor(width * .10)) / 3) * 3;
    // Reject frames whose face is already too close to the edges to make a
    // normal album portrait. They are the source of the “big head” result:
    // there is not enough surrounding image to keep the face at a consistent
    // size in the final 3:4 crop.
    const canUseAlbumScale = faceWidthPixels / maxCropWidth <= ALBUM_FACE_RATIO;
    // A candidate is only eligible when a 3:4 crop can contain the complete
    // detected face with a little safety margin. This rejects side faces and
    // frames where the detector found only a cheek/forehead beside an arm.
    return candidate.score >= .50 && candidate.frontal >= .40 && candidate.featureScore >= .12 && candidate.faceConfidence >= .46 && candidate.colorPixels >= 80 && faceWidthPixels >= Math.max(120, width * .16) && maxCropWidth >= 120 && canUseAlbumScale && hasCompleteCropRoom(candidate);
  });
  const bestClarity = structural.reduce((best, candidate) => Math.max(best, candidate.clarity), 0);
  const clarityFloor = bestClarity ? Math.max(.10, bestClarity * .58) : 0;
  const eligible = structural.filter(candidate => candidate.clarity >= clarityFloor);
  const jpgs = eligible.filter(candidate => /^\.?jpe?g$/i.test(candidate.ext));
  const gifs = eligible.filter(candidate => /^\.?gif$/i.test(candidate.ext));
  const other = eligible.filter(candidate => !/^\.?jpe?g$/i.test(candidate.ext) && !/^\.?gif$/i.test(candidate.ext));
  const strongJpgs = jpgs.filter(candidate => candidate.faceConfidence >= .50 && candidate.featureScore >= .14 && candidate.clarity >= .10);
  const strongGifs = gifs.filter(candidate => candidate.faceConfidence >= .48 && candidate.featureScore >= .14 && candidate.clarity >= .10);
  const widthGap = (a: FaceCandidate, b: FaceCandidate) => Math.abs(Math.log((a.faceWidth * a.image.naturalWidth) / (b.faceWidth * b.image.naturalWidth)));
  const pairValue = (a: FaceCandidate, b: FaceCandidate) => (a.score + b.score) / 2 + (a.clarity + b.clarity) * .16 + (a.frontal + b.frontal) * .14 + (a.featureScore + b.featureScore) * .16 - widthGap(a, b) * .22;
  const bestPair = (pool: FaceCandidate[]) => {
    if (pool.length <= 1) return [];
    let best: FaceCandidate[] | null = null, value = -Infinity;
    for (let i = 0; i < pool.length; i++) for (let j = i + 1; j < pool.length; j++) {
      if (pool[i].asset.index === pool[j].asset.index && Math.abs(pool[i].frameIndex - pool[j].frameIndex) < 3) continue;
      if (widthGap(pool[i], pool[j]) > Math.log(1.5)) continue;
      const score = pairValue(pool[i], pool[j]);
      if (score > value) { value = score; best = [pool[i], pool[j]]; }
    }
    return best || [];
  };
  // JPG is always the first choice. If there are two usable JPGs, use those.
  // If there is only one, a qualified GIF frame may supply the second view;
  // this preserves the JPG while still enforcing the two-image requirement.
  if (strongJpgs.length >= 2) return bestPair(strongJpgs);
  const jpgFallback = candidates.filter(candidate => {
    const width = candidate.image.naturalWidth, height = candidate.image.naturalHeight;
    const faceWidthPixels = candidate.faceWidth * width / 160;
    const maxCropWidth = Math.floor(Math.min(width, (Math.floor(height * .96) - Math.ceil(height * .14)) * .75, Math.ceil(width * .98) - Math.floor(width * .10)) / 3) * 3;
    // A JPG is preferred only when it actually resembles a frontal face. A
    // warm-colored scan by itself is not enough; otherwise arms and placenta
    // images win simply because they contain more orange pixels.
    return /^\.?jpe?g$/i.test(candidate.ext) && candidate.colorPixels >= 80 && candidate.faceWidth >= 18 && candidate.score >= .42 && candidate.frontal >= .36 && candidate.featureScore >= .10 && candidate.faceConfidence >= .42 && candidate.clarity >= .08 && faceWidthPixels >= Math.max(110, width * .12) && maxCropWidth >= 120 && hasCompleteCropRoom(candidate);
  }).map(candidate => {
    // A qualified JPG is the preferred source. Keep its detected facial
    // position; cropPlan will calculate the consistent margin around it.
    return {
      ...candidate,
      cx: candidate.cx,
      cy: candidate.cy,
    };
  });
  if (jpgFallback.length >= 2) return bestPair(jpgFallback);
  if (strongJpgs.length === 1 && strongGifs.length) return bestPair([...strongJpgs, ...strongGifs]);
  if (jpgFallback.length === 1 && strongGifs.length) return bestPair([...jpgFallback, ...strongGifs]);
  if (strongGifs.length >= 2) return bestPair(strongGifs);
  if (other.length >= 2) return bestPair(other);
  // If no JPG passes the face test, use GIF frames converted to standalone
  // JPGs. The same frontal-face and clarity gates apply to GIFs.
  const gifFallback = candidates.filter(candidate => {
    const width = candidate.image.naturalWidth, height = candidate.image.naturalHeight;
    const faceWidthPixels = candidate.faceWidth * width / 160;
    const maxCropWidth = Math.floor(Math.min(width, (Math.floor(height * .96) - Math.ceil(height * .14)) * .75, Math.ceil(width * .98) - Math.floor(width * .10)) / 3) * 3;
    return /^\.?gif$/i.test(candidate.ext) && candidate.score >= .42 && candidate.frontal >= .36 && candidate.featureScore >= .10 && candidate.faceConfidence >= .42 && candidate.clarity >= .08 && candidate.colorPixels >= 80 && faceWidthPixels >= Math.max(110, width * .14) && maxCropWidth >= 120 && faceWidthPixels / maxCropWidth <= .72 && hasCompleteCropRoom(candidate);
  });
  if (gifFallback.length >= 2) return bestPair(gifFallback);
  // Never manufacture a result from a warm-colored blob. An incomplete face
  // is worse than a visible failure because it would be saved as a valid baby
  // album image.
  return [];
}

function maxCropWidthFor(candidate: FaceCandidate) {
  const width = candidate.image.naturalWidth, height = candidate.image.naturalHeight;
  const safeTop = Math.ceil(height * .14), safeBottom = Math.floor(height * .96), safeLeft = Math.floor(width * .10), safeRight = Math.ceil(width * .98);
  return Math.floor(Math.min(width, (safeBottom - safeTop) * .75, safeRight - safeLeft) / 3) * 3;
}

function targetCropWidthFor(candidate: FaceCandidate) {
  const faceWidthPixels = candidate.faceWidth * candidate.image.naturalWidth / 160;
  return Math.floor(faceWidthPixels / ALBUM_FACE_RATIO / 3) * 3;
}

function cropPlan(candidate: FaceCandidate, cropWidth: number) {
  const image = candidate.image, width = image.naturalWidth, height = image.naturalHeight;
  const cropHeight = cropWidth * 4 / 3;
  const boxWidth = Math.max(1, candidate.faceRight - candidate.faceLeft);
  const boxHeight = Math.max(1, candidate.faceBottom - candidate.faceTop);
  const marginX = boxWidth * .12;
  const marginY = boxHeight * .12;
  const faceLeft = candidate.faceLeft - marginX;
  const faceRight = candidate.faceRight + marginX;
  const faceTop = candidate.faceTop - marginY;
  const faceBottom = candidate.faceBottom + marginY;
  const leftMin = Math.max(0, faceRight - cropWidth);
  const leftMax = Math.min(width - cropWidth, faceLeft);
  const topMin = Math.max(0, faceBottom - cropHeight);
  const topMax = Math.min(height - cropHeight, faceTop);
  if (leftMin > leftMax || topMin > topMax) return null;
  const left = Math.round(clamp(candidate.cx - cropWidth / 2, leftMin, leftMax));
  const top = Math.round(clamp(candidate.cy - cropHeight * .38, topMin, topMax));
  return { left, top, cropHeight };
}

async function crop(candidate: FaceCandidate, commonWidth: number): Promise<Crop> {
  const image = candidate.image, width = image.naturalWidth, height = image.naturalHeight;
  const maxCropWidth = Math.floor(Math.min(width, height * .75) / 3) * 3;
  // The source crop width varies with the detected face, but every result is
  // rendered to the same output width. This keeps the baby's face the same
  // size across the two album images instead of letting one become a close-up.
  const cropWidth = Math.min(targetCropWidthFor(candidate), maxCropWidth);
  const cropHeight = cropWidth * 4 / 3;
  if (cropWidth < 3) throw new Error("影像分辨率不足，无法裁成 3:4。");
  const plan = cropPlan(candidate, cropWidth);
  if (!plan) throw new Error("完整人脸周围没有足够的连续影像范围，已跳过该帧。");
  const { left, top } = plan;
  const canvas = document.createElement("canvas"); canvas.width = commonWidth; canvas.height = commonWidth * 4 / 3;
  const ctx = canvas.getContext("2d"); if (!ctx) throw new Error("无法生成截图。");
  ctx.drawImage(image, left, top, cropWidth, cropHeight, 0, 0, canvas.width, canvas.height);
  const blob = await new Promise<Blob>((resolve, reject) => canvas.toBlob(value => value ? resolve(value) : reject(new Error("PNG 编码失败。")), "image/png"));
  return { dataUrl: canvas.toDataURL("image/png"), width: cropWidth, height: cropHeight, bytes: blob.size };
}

function decodeBase64(value: string) { return value.slice(value.indexOf(",") + 1); }

async function downloadZip(code: string, name: string, crops: Crop[]) {
  if (!window.JSZip) throw new Error("ZIP 组件尚未加载，请刷新页面后重试。");
  const zip = new window.JSZip(), folder = zip.folder(`${code} ${name}`);
  if (!folder) throw new Error("无法创建输出文件夹。");
  crops.forEach((image, index) => folder.file(`${index + 1}.png`, decodeBase64(image.dataUrl), { base64: true }));
  const blob = await zip.generateAsync({ type: "blob", compression: "STORE" });
  const url = URL.createObjectURL(blob), link = document.createElement("a");
  link.href = url; link.download = `${code} ${name}.zip`; link.style.display = "none";
  document.body.appendChild(link); link.click(); link.remove(); window.setTimeout(() => URL.revokeObjectURL(url), 60000);
}

async function saveCaseFolder(root: SaveDirectory, code: string, name: string, crops: Crop[]) {
  const permission = root.queryPermission ? await root.queryPermission({ mode: "readwrite" }) : "granted";
  if (permission !== "granted") throw new Error("保存目录权限已失效，请重新点击“选择保存目录”并允许写入。");
  const folder = await root.getDirectoryHandle(`${code} ${name}`, { create: true });
  for (let i = 0; i < crops.length; i++) {
    const handle = await folder.getFileHandle(`${i + 1}.png`, { create: true });
    const writer = await handle.createWritable();
    const raw = atob(decodeBase64(crops[i].dataUrl));
    const bytes = new Uint8Array(raw.length); for (let j = 0; j < raw.length; j++) bytes[j] = raw.charCodeAt(j);
    await writer.write(bytes); await writer.close();
  }
}

async function searchImages(code: string, name: string): Promise<Asset[]> {
  const response = await fetch("/api/search", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ code, name }) });
  const raw = await response.text();
  let result: { assets?: Asset[]; error?: string };
  try { result = JSON.parse(raw) as { assets?: Asset[]; error?: string }; }
  catch { throw new Error(`查询服务返回了网页而不是结果（HTTP ${response.status}）。请刷新页面后重试。`); }
  if (!response.ok) throw new Error(result.error || "查询影像失败，请检查网络。");
  return result.assets ?? [];
}

async function runPhoto(file: File | undefined, manualCode: string, manualName: string, setStatus: (text: string) => void, saveDirectory?: SaveDirectory): Promise<CaseResult> {
  let fields: { code: string | null; name: string | null } = { code: null, name: null };
  if (file) {
    if (file.size > 20 * 1024 * 1024) throw new Error("报告照片超过 20 MB，请换一张较小的照片。");
    if (!window.Tesseract) throw new Error("中文识别组件还没加载好，请刷新后再试。");
    setStatus("正在本机识别报告单…");
    const worker = await window.Tesseract.createWorker("chi_sim", 1, {
      langPath: "/ocr/tessdata", workerPath: "/ocr/worker.min.js", corePath: "/ocr/tesseract-core-simd-lstm.wasm.js", gzip: true,
    });
    try { fields = parseReport((await worker.recognize(file)).data.text); }
    finally { await worker.terminate(); }
  }
  const code = manualCode.trim() || fields.code;
  const name = manualName.trim() || fields.name;
  if (!code || !/^\d{6,20}$/.test(code)) throw new Error("请填写正确的 6–20 位超声号，或上传清晰的报告照片识别。");
  if (!name) throw new Error("请填写彩超报告单上的中文姓名，用于命名保存文件夹。");
  setStatus(file ? `已识别/填写 ${code}，正在查询影像…` : `正在按超声号 ${code} 查询影像…`);
  const assets = await searchImages(code, name);
  if (!assets.length) throw new Error("没有找到这个超声号的影像，请检查报告单或稍后重试。");
  setStatus(`找到 ${assets.length} 张影像，先筛选原始 JPG；只有 JPG 不可用时才逐帧转换 GIF…`);
  const candidates: FaceCandidate[] = [];
  for (const asset of assets) { try { candidates.push(...await snapshot(asset)); } catch { /* Skip unreadable images */ } }
  const selected = chooseFaceCandidates(candidates);
  if (selected.length !== 2) throw new Error("没有找到两张都能完整显示小朋友正脸的彩色影像；不完整、遮挡或角度不合适的图片已跳过，本组不会生成残缺截图。");
  const crops: Crop[] = [];
  const commonWidth = Math.floor(Math.min(...selected.map(candidate => Math.min(targetCropWidthFor(candidate), Math.floor(Math.min(candidate.image.naturalWidth, candidate.image.naturalHeight * .75) / 3) * 3))) / 3) * 3;
  if (commonWidth < 120) throw new Error("合格正脸的可用范围太小，无法生成统一大小的相册照片。");
  for (const candidate of selected) crops.push(await crop(candidate, commonWidth));
  setStatus("正脸截图已生成，正在保存到本机…");
  if (saveDirectory) {
    try { await saveCaseFolder(saveDirectory, code, name, crops); setStatus(`完成。已保存到文件夹“${code} ${name}”。`); }
    catch (error) { setStatus(error instanceof Error ? `${error.message} 已改为下载 ZIP。` : "保存目录不可用，已改为下载 ZIP。"); await downloadZip(code, name, crops); }
  } else await downloadZip(code, name, crops);
  if (!saveDirectory) setStatus(`完成。已下载“${code} ${name}.zip”，解压后包含 ${crops.length === 1 ? "一张合格人像图" : "两张合格人像图"}。`);
  return { code, name, crops };
}

export default function AutoFlow({ signOutHref }: { signOutHref: string }) {
  const [status,setStatus]=useState("选好彩超报告单照片后自动处理；报告照片在浏览器本机识别。");
  const [error,setError]=useState(false);
  const [result,setResult]=useState<CaseResult|null>(null);
  const [batchResults,setBatchResults]=useState<CaseResult[]>([]);
  const [batchFailures,setBatchFailures]=useState<BatchFailure[]>([]);
  const [batchRows,setBatchRows]=useState(Array.from({length:10},()=>({code:"",name:""})));
  const [saveDirectory,setSaveDirectory]=useState<SaveDirectory|null>(null);
  const [busy,setBusy]=useState(false);
  const [batchBusy,setBatchBusy]=useState(false);
  const input=useRef<HTMLInputElement>(null);
  const batchInput=useRef<HTMLTextAreaElement>(null);
  const codeInput=useRef<HTMLInputElement>(null);
  const nameInput=useRef<HTMLInputElement>(null);

  async function handle(file?:File){
    if(busy||(!file&&!codeInput.current?.value.trim()))return;setBusy(true);setError(false);
    try{setResult(await runPhoto(file,codeInput.current?.value??"",nameInput.current?.value??"",setStatus,saveDirectory??undefined))}
    catch(reason){setError(true);setStatus(reason instanceof Error?reason.message:"处理失败，请重新拍摄报告单后再试。")}
    finally{setBusy(false);if(input.current)input.current.value=""}
  }

  async function runBatch() {
    if (batchBusy || busy) return;
    const items = batchRows.filter(row => row.code.trim() && row.name.trim()).map(row => ({ code: row.code.trim(), name: row.name.trim() }));
    if (!items.length) { setStatus("请按每行“超声号 姓名”填写，最多 10 行。"); return; }
    setBatchBusy(true); setBatchResults([]); setBatchFailures([]); setError(false);
    const done: CaseResult[] = [];
    const failed: BatchFailure[] = [];
    for (let i = 0; i < items.length; i++) {
      const item = items[i];
      setStatus(`正在处理第 ${i + 1}/${items.length} 组：${item.code}…`);
      try {
        done.push(await runPhoto(undefined, item.code, item.name, setStatus, saveDirectory??undefined));
        setBatchResults([...done]);
      } catch (reason) {
        const message = reason instanceof Error ? reason.message : "本组处理失败。";
        failed.push({ code: item.code, name: item.name, message });
        setBatchFailures([...failed]);
      }
      if (i < items.length - 1) await new Promise(resolve => setTimeout(resolve, BATCH_QUERY_GAP_MS));
    }
    setError(failed.length > 0);
    setStatus(`批量处理完成：成功 ${done.length} 组，失败 ${failed.length} 组。失败的组不会影响其他组。`);
    setBatchBusy(false);
  }

  return <main className="mx-auto min-h-screen w-full max-w-5xl px-5 py-8 text-slate-900 sm:px-8 sm:py-12">
    <header className="mb-8 flex items-start justify-between gap-5">
      <div><p className="mb-2 text-sm font-semibold tracking-wide text-teal-700">四维影像整理</p><h1 className="text-3xl font-semibold tracking-tight">上传报告，自动截正脸</h1><p className="mt-3 max-w-xl text-slate-600">先核对超声号和报告姓名，再从彩色影像中筛选完整人脸，自动居中裁成 3:4。</p></div>
      <a href={signOutHref} target="_top" className="shrink-0 rounded-lg border border-slate-200 px-3 py-2 text-sm text-slate-600 hover:bg-slate-50">退出</a>
    </header>
    <section className="rounded-2xl border border-slate-200 bg-white p-5 shadow-sm sm:p-7">
      <h2 className="text-xl font-semibold">输入 1–10 组超声号和姓名</h2>
      <p className="mt-2 text-sm text-slate-600">每一行对应一个小朋友；空行会跳过。每组都会生成两张正脸预览。</p>
      <div className="mt-4 space-y-2">{batchRows.map((row,i)=><div key={i} className="grid grid-cols-[3rem_1fr_1fr] gap-2"><span className="pt-2 text-sm text-slate-500">{i+1}</span><input value={row.code} onChange={e=>setBatchRows(rows=>rows.map((r,j)=>j===i?{...r,code:e.target.value}:r))} inputMode="numeric" placeholder="超声号" className="rounded-lg border border-slate-300 px-3 py-2"/><input value={row.name} onChange={e=>setBatchRows(rows=>rows.map((r,j)=>j===i?{...r,name:e.target.value}:r))} placeholder="报告单中文姓名" className="rounded-lg border border-slate-300 px-3 py-2"/></div>)}</div>
      <div className="mt-4 flex flex-wrap gap-3"><button type="button" disabled={batchBusy} onClick={runBatch} className="rounded-lg bg-teal-700 px-4 py-2.5 text-sm font-semibold text-white disabled:opacity-60">{batchBusy?"正在批量处理…":"开始批量处理"}</button><button type="button" onClick={async()=>{const picker=(window as Window & { showDirectoryPicker?: (options?: {mode?:"read"|"readwrite"}) => Promise<SaveDirectory> }).showDirectoryPicker;if(!picker){setStatus("请使用最新版 Chrome 或 Edge 才能直接保存文件夹。");return}try{const handle=await picker({mode:"readwrite"});if(handle.requestPermission){const permission=await handle.requestPermission({mode:"readwrite"});if(permission!=="granted")throw new Error("没有获得写入权限")}setSaveDirectory(handle);setStatus("已选择保存根目录，批量结果会直接写入文件夹。") }catch(reason){setSaveDirectory(null);setStatus(reason instanceof Error?reason.message:"已取消选择保存目录，结果仍可下载 ZIP。")}}} className="rounded-lg border border-teal-700 px-4 py-2.5 text-sm font-semibold text-teal-800">{saveDirectory?`保存目录：${saveDirectory.name}`:"选择本机保存目录"}</button></div>
      <p className="mt-2 text-sm text-slate-500">请选择普通空文件夹或桌面上的新建文件夹，不要选择系统目录；程序会在里面自动创建“超声号 姓名”子文件夹。</p>
      <p aria-live="polite" className={`mt-5 rounded-lg px-4 py-3 text-sm ${error?"bg-red-50 text-red-700":"bg-slate-50 text-slate-700"}`}>{status}</p>
    </section>
    <section className="mt-6 rounded-2xl border border-slate-200 bg-white p-5 shadow-sm sm:p-7">
      <h2 className="text-xl font-semibold">批量处理（最多 10 组）</h2>
      <p className="mt-2 text-sm text-slate-600">每行填写“超声号 姓名”，例如：20260925012 刘佳。每组会单独查询并下载 ZIP，同时保留页面预览。</p>
      <textarea ref={batchInput} rows={6} disabled={batchBusy} placeholder="20260925012 刘佳\n20260927014 赵香雨" className="mt-4 w-full rounded-lg border border-slate-300 px-3 py-2.5 text-base outline-none focus:border-teal-600 focus:ring-2 focus:ring-teal-100" />
      <button type="button" disabled={batchBusy||busy} onClick={runBatch} className="mt-3 rounded-lg bg-teal-700 px-4 py-2.5 text-sm font-semibold text-white disabled:opacity-60">{batchBusy?"正在批量处理…":"开始批量处理"}</button>
      {batchFailures.length>0&&<div className="mt-6 space-y-2 rounded-xl border border-red-200 bg-red-50 p-4"><h3 className="font-semibold text-red-800">失败的组（其他组仍会继续处理）</h3>{batchFailures.map(item=><p key={item.code} className="text-sm text-red-700">{item.code} · {item.name}：{item.message}</p>)}</div>}
      {batchResults.length>0&&<div className="mt-6 space-y-6">{batchResults.map(item=><article key={item.code} className="rounded-xl border border-slate-200 p-4"><h3 className="font-semibold">{item.code} · {item.name}</h3><div className="mt-3 grid gap-4 sm:grid-cols-2">{item.crops.map((c,i)=><figure key={i} className="overflow-hidden rounded-lg border border-slate-200 bg-black"><img src={c.dataUrl} alt={`${item.code} 自动截图 ${i+1}`} className="aspect-[3/4] w-full object-contain"/><figcaption className="bg-white px-3 py-2 text-xs text-slate-600">截图 {i+1} · {c.width}×{c.height} · {Math.round(c.bytes/1024)} KB</figcaption></figure>)}</div></article>)}</div>}
    </section>
    {result&&<section className="mt-6 rounded-2xl border border-slate-200 bg-white p-5 shadow-sm sm:p-7">
      <h2 className="text-xl font-semibold">自动截图结果</h2>
      <p className="mt-2 text-sm text-slate-600">{result.code} · {result.name} · 已打包下载。解压后文件夹名为“{result.code} {result.name}”，含 {result.crops.length===1?"1.png":"1.png 和 2.png"}。</p>
      <div className="mt-5 grid grid-cols-1 gap-4 sm:grid-cols-2">{result.crops.map((c,i)=><figure key={i} className="overflow-hidden rounded-xl border border-slate-200 bg-black"><img src={c.dataUrl} alt={`自动截图 ${i+1}`} className="aspect-[3/4] w-full object-contain"/><figcaption className="bg-white px-3 py-2 text-sm text-slate-600">{c.width}×{c.height} · {Math.round(c.bytes/1024)} KB</figcaption></figure>)}</div>
      {result.crops.some(c=>c.bytes<100*1024)&&<p className="mt-3 text-sm text-amber-700">原片分辨率有限，文件可能低于 100 KB；未放大或填充。</p>}
    </section>}
    <footer className="mt-6 text-sm leading-6 text-slate-500">报告照片在浏览器本机识别，不上传保存。查询时会将超声号和报告姓名用于核对医院影像档案；不一致则停止下载。影像处理后由浏览器下载 ZIP，不在网站留存。截图为原片像素裁切，不修图、不生成新影像。</footer>
  </main>
}

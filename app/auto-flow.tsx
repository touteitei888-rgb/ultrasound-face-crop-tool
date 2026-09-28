"use client";

import { decompressFrames, parseGIF } from "gifuct-js";
import { useRef, useState } from "react";

type Asset = { index: number; mime: string; ext: string; bytes: string };
type Crop = { dataUrl: string; width: number; height: number; bytes: number };
type Worker = { recognize: (image: File) => Promise<{ data: { text: string; confidence: number } }>; terminate: () => Promise<void> };
type FaceCandidate = ReturnType<typeof analyze> & { image: HTMLImageElement; ext: string; asset: Asset; frameIndex: number };

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
const ALBUM_FACE_RATIO = .68;

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
  const empty = { score: 0, frontal: 0, cx: width * .5, cy: height * .42, faceWidth: 0, faceHeight: 0, colorPixels: warmCount, clarity, featureScore: 0, faceConfidence: 0 };
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
  const faceBottom = Math.min(largest.maxY, Math.round(largest.minY + (largest.maxY - largest.minY) * .57));
  let sx = 0, sy = 0, pixelsInFace = 0, faceMinX = sw, faceMaxX = 0;
  for (const p of largest.points) {
    const x = p % sw, y = Math.floor(p / sw); if (y > faceBottom) continue;
    sx += x; sy += y; pixelsInFace++; faceMinX = Math.min(faceMinX, x); faceMaxX = Math.max(faceMaxX, x);
  }
  if (pixelsInFace < 80 || faceMaxX - faceMinX < 18) return empty;
  const cx = sx / pixelsInFace, cy = sy / pixelsInFace, faceWidth = faceMaxX - faceMinX + 1, faceHeight = faceBottom - largest.minY + 1;
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
  let darkLeft = 0, darkRight = 0;
  const featureTop = Math.round(largest.minY + faceHeight * .08), featureBottom = Math.round(largest.minY + faceHeight * .68);
  const featureLeft = Math.round(faceMinX + faceWidth * .14), featureRight = Math.round(faceMaxX - faceWidth * .14);
  for (let y = featureTop; y <= featureBottom; y++) for (let x = featureLeft; x <= featureRight; x++) {
    const i = (y * sw + x) * 4, luminance = pixels[i] * .299 + pixels[i + 1] * .587 + pixels[i + 2] * .114;
    if (luminance >= 62) continue;
    let warmNeighbors = 0;
    for (let ny = Math.max(0, y - 2); ny <= Math.min(sh - 1, y + 2); ny++) for (let nx = Math.max(0, x - 2); nx <= Math.min(sw - 1, x + 2); nx++) warmNeighbors += largestMask[ny * sw + nx];
    if (warmNeighbors < 4) continue;
    if (x < cx) darkLeft++; else darkRight++;
  }
  const darkTotal = darkLeft + darkRight;
  const darkBalance = darkTotal ? 2 * Math.min(darkLeft, darkRight) / darkTotal : 0;
  const darkPresence = Math.min(1, darkTotal / 32);
  const featureScore = darkBalance * darkPresence;
  const faceConfidence = frontal * .40 + coverage * .16 + balance * .10 + clarity * .16 + featureScore * .18;
  return { score: frontal * .48 + coverage * .20 + balance * .10 + clarity * .10 + featureScore * .12, frontal, cx: cx / sw * width, cy: cy / sh * height, faceWidth, faceHeight, colorPixels: warmCount, clarity, featureScore, faceConfidence };
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
    return candidate.score >= .50 && candidate.frontal >= .40 && candidate.colorPixels >= 80 && faceWidthPixels >= Math.max(120, width * .16) && maxCropWidth >= 120 && canUseAlbumScale;
  });
  const bestClarity = structural.reduce((best, candidate) => Math.max(best, candidate.clarity), 0);
  const clarityFloor = bestClarity ? Math.max(.10, bestClarity * .58) : 0;
  const eligible = structural.filter(candidate => candidate.clarity >= clarityFloor);
  const jpgs = eligible.filter(candidate => /^\.?jpe?g$/i.test(candidate.ext));
  const gifs = eligible.filter(candidate => /^\.?gif$/i.test(candidate.ext));
  const other = eligible.filter(candidate => !/^\.?jpe?g$/i.test(candidate.ext) && !/^\.?gif$/i.test(candidate.ext));
  const strongJpgs = jpgs.filter(candidate => candidate.faceConfidence >= .50 && candidate.clarity >= .10);
  const strongGifs = gifs.filter(candidate => candidate.faceConfidence >= .44 && candidate.clarity >= .10);
  const widthGap = (a: FaceCandidate, b: FaceCandidate) => Math.abs(Math.log((a.faceWidth * a.image.naturalWidth) / (b.faceWidth * b.image.naturalWidth)));
  const pairValue = (a: FaceCandidate, b: FaceCandidate) => (a.score + b.score) / 2 + (a.clarity + b.clarity) * .12 - widthGap(a, b) * .22;
  const bestPair = (pool: FaceCandidate[]) => {
    if (pool.length <= 1) return pool.slice();
    let best: typeof eligible | null = null, value = -Infinity;
    for (let i = 0; i < pool.length; i++) for (let j = i + 1; j < pool.length; j++) {
      if (pool[i].asset.index === pool[j].asset.index && Math.abs(pool[i].frameIndex - pool[j].frameIndex) < 3) continue;
      if (widthGap(pool[i], pool[j]) > Math.log(1.5)) continue;
      const score = pairValue(pool[i], pool[j]);
      if (score > value) { value = score; best = [pool[i], pool[j]]; }
    }
    return best || [pool.slice().sort((a, b) => b.score - a.score)[0]];
  };
  // JPG is always the first choice. Do not add a GIF beside a usable JPG:
  // GIF is only a fallback for cases where all JPGs are unusable.
  if (strongJpgs.length) return bestPair(strongJpgs);
  const jpgFallback = candidates.filter(candidate => {
    const width = candidate.image.naturalWidth, height = candidate.image.naturalHeight;
    const faceWidthPixels = candidate.faceWidth * width / 160;
    const maxCropWidth = Math.floor(Math.min(width, (Math.floor(height * .96) - Math.ceil(height * .14)) * .75, Math.ceil(width * .98) - Math.floor(width * .10)) / 3) * 3;
    return /^\.?jpe?g$/i.test(candidate.ext) && candidate.colorPixels >= 80 && candidate.faceWidth >= 12 && candidate.score >= .20 && candidate.frontal >= .10 && candidate.clarity >= .03 && faceWidthPixels >= Math.max(70, width * .08) && maxCropWidth >= 120;
  }).map(candidate => {
    // A color JPG is the preferred source. Its whole-frame layout is stable,
    // so use one standard face scale instead of letting a noisy mask turn it
    // into a close-up or rejecting it altogether.
    const width = candidate.image.naturalWidth, height = candidate.image.naturalHeight;
    return {
      ...candidate,
      // The raw JPG contains the complete ultrasound layout. Keep the detected
      // facial position when it is plausible, then give the face only a small
      // consistent margin instead of including the arm/placenta region.
      cx: candidate.cx > width * .28 && candidate.cx < width * .72 ? candidate.cx : width * .47,
      cy: candidate.cy > height * .24 && candidate.cy < height * .68 ? candidate.cy : height * .45,
      faceWidth: clamp(candidate.faceWidth, 34, 40),
      faceHeight: clamp(candidate.faceHeight, 34, 40),
      frontal: Math.max(candidate.frontal, .28),
      score: Math.max(candidate.score, .42),
      faceConfidence: Math.max(candidate.faceConfidence, .32),
    };
  });
  if (jpgFallback.length) return bestPair(jpgFallback);
  if (strongGifs.length) return bestPair(strongGifs);
  if (other.length) return bestPair(other);
  // Some valid 3D ultrasound faces have soft edges or only a brief frontal
  // view, so keep a controlled rescue path instead of returning nothing. It
  // still excludes monochrome scans and frames that cannot fit the common
  // album face scale.
  const rescue = candidates.filter(candidate => {
    const width = candidate.image.naturalWidth, height = candidate.image.naturalHeight;
    const faceWidthPixels = candidate.faceWidth * width / 160;
    const maxCropWidth = Math.floor(Math.min(width, (Math.floor(height * .96) - Math.ceil(height * .14)) * .75, Math.ceil(width * .98) - Math.floor(width * .10)) / 3) * 3;
    return !/^\.?jpe?g$/i.test(candidate.ext) && candidate.score >= .38 && candidate.frontal >= .28 && candidate.clarity >= .06 && candidate.colorPixels >= 80 && faceWidthPixels >= Math.max(100, width * .14) && maxCropWidth >= 120 && faceWidthPixels / maxCropWidth <= .74;
  });
  if (rescue.length) return bestPair(rescue);
  // Do not export a clearly non-face JPG just to fill the second slot.
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

async function crop(candidate: FaceCandidate, commonWidth: number): Promise<Crop> {
  const image = candidate.image, width = image.naturalWidth, height = image.naturalHeight;
  const safeTop = Math.ceil(height * .14), safeBottom = Math.floor(height * .96), safeLeft = Math.floor(width * .10), safeRight = Math.ceil(width * .98);
  const maxCropWidth = Math.floor(Math.min(width, (safeBottom - safeTop) * .75, safeRight - safeLeft) / 3) * 3;
  // The source crop width varies with the detected face, but every result is
  // rendered to the same output width. This keeps the baby's face the same
  // size across the two album images instead of letting one become a close-up.
  const cropWidth = Math.min(targetCropWidthFor(candidate), maxCropWidth);
  const cropHeight = cropWidth * 4 / 3;
  if (cropWidth < 3) throw new Error("影像分辨率不足，无法裁成 3:4。");
  const cx = clamp(candidate.cx, safeLeft + cropWidth / 2, safeRight - cropWidth / 2);
  const top = Math.round(clamp(candidate.cy - cropHeight * .38, safeTop, safeBottom - cropHeight));
  const left = Math.round(clamp(cx - cropWidth / 2, safeLeft, safeRight - cropWidth));
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

async function searchImages(code: string, name: string): Promise<Asset[]> {
  const response = await fetch("/api/search", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ code, name }) });
  const result = await response.json() as { assets?: Asset[]; error?: string };
  if (!response.ok) throw new Error(result.error || "查询影像失败，请检查网络。");
  return result.assets ?? [];
}

async function runPhoto(file: File | undefined, manualCode: string, manualName: string, setStatus: (text: string) => void, setResult: (value: { code: string; name: string; crops: Crop[] } | null) => void) {
  setResult(null);
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
  if (!selected.length) throw new Error("没有找到能完整显示小朋友人脸的彩色影像；黑白平扫图或角度不合适的图片已跳过。");
  const crops: Crop[] = [];
  const commonWidth = Math.floor(Math.min(...selected.map(candidate => Math.min(targetCropWidthFor(candidate), maxCropWidthFor(candidate)))) / 3) * 3;
  if (commonWidth < 120) throw new Error("合格正脸的可用范围太小，无法生成统一大小的相册照片。");
  for (const candidate of selected) crops.push(await crop(candidate, commonWidth));
  setStatus("正脸截图已生成，正在保存到本机…");
  await downloadZip(code, name, crops);
  setResult({ code, name, crops });
  setStatus(`完成。已下载“${code} ${name}.zip”，解压后包含 ${crops.length === 1 ? "一张合格人像图" : "两张合格人像图"}。`);
}

export default function AutoFlow({ signOutHref }: { signOutHref: string }) {
  const [status,setStatus]=useState("选好彩超报告单照片后自动处理；报告照片在浏览器本机识别。");
  const [error,setError]=useState(false);
  const [result,setResult]=useState<{code:string;name:string;crops:Crop[]}|null>(null);
  const [busy,setBusy]=useState(false);
  const input=useRef<HTMLInputElement>(null);
  const codeInput=useRef<HTMLInputElement>(null);
  const nameInput=useRef<HTMLInputElement>(null);

  async function handle(file?:File){
    if(busy||(!file&&!codeInput.current?.value.trim()))return;setBusy(true);setError(false);
    try{await runPhoto(file,codeInput.current?.value??"",nameInput.current?.value??"",setStatus,setResult)}
    catch(reason){setError(true);setStatus(reason instanceof Error?reason.message:"处理失败，请重新拍摄报告单后再试。")}
    finally{setBusy(false);if(input.current)input.current.value=""}
  }

  return <main className="mx-auto min-h-screen w-full max-w-3xl px-5 py-8 text-slate-900 sm:px-8 sm:py-12">
    <header className="mb-8 flex items-start justify-between gap-5">
      <div><p className="mb-2 text-sm font-semibold tracking-wide text-teal-700">四维影像整理</p><h1 className="text-3xl font-semibold tracking-tight">上传报告，自动截正脸</h1><p className="mt-3 max-w-xl text-slate-600">先核对超声号和报告姓名，再从彩色影像中筛选完整人脸，自动居中裁成 3:4。</p></div>
      <a href={signOutHref} target="_top" className="shrink-0 rounded-lg border border-slate-200 px-3 py-2 text-sm text-slate-600 hover:bg-slate-50">退出</a>
    </header>
    <section className="rounded-2xl border border-slate-200 bg-white p-5 shadow-sm sm:p-7">
      <input ref={input} type="file" accept="image/jpeg,image/png,image/webp" className="sr-only" onChange={event=>handle(event.target.files?.[0])}/>
      <button type="button" disabled={busy} onClick={()=>input.current?.click()} onDragOver={event=>event.preventDefault()} onDrop={event=>{event.preventDefault();handle(event.dataTransfer.files?.[0])}} className="flex min-h-48 w-full flex-col items-center justify-center rounded-xl border-2 border-dashed border-slate-300 bg-slate-50 px-6 py-8 text-center transition hover:border-teal-600 hover:bg-teal-50/40 disabled:cursor-wait disabled:opacity-60">
        <span className="mb-3 grid h-12 w-12 place-items-center rounded-full bg-teal-100 text-2xl text-teal-800">＋</span>
        <span className="text-lg font-semibold">{busy?"正在自动处理…":"选择或拖入彩超报告单照片"}</span>
        <span className="mt-2 text-sm text-slate-500">识别成功后自动查询和截图；识别失败时可手动填写信息继续</span>
      </button>
      <div className="mt-5 grid gap-4 sm:grid-cols-2">
        <label className="block text-sm font-medium text-slate-700">超声号<input ref={codeInput} inputMode="numeric" autoComplete="off" placeholder="例如：20260927002" className="mt-1.5 w-full rounded-lg border border-slate-300 px-3 py-2.5 text-base outline-none focus:border-teal-600 focus:ring-2 focus:ring-teal-100"/></label>
        <label className="block text-sm font-medium text-slate-700">报告单上的中文姓名<input ref={nameInput} autoComplete="off" placeholder="用于命名保存文件夹" className="mt-1.5 w-full rounded-lg border border-slate-300 px-3 py-2.5 text-base outline-none focus:border-teal-600 focus:ring-2 focus:ring-teal-100"/></label>
      </div>
      <button type="button" disabled={busy} onClick={()=>handle()} className="mt-3 rounded-lg bg-teal-700 px-4 py-2.5 text-sm font-semibold text-white transition hover:bg-teal-800 disabled:cursor-wait disabled:opacity-60">按填写的信息查询并自动截图</button>
      <p className="mt-2 text-sm text-slate-500">如果照片识别失败，请手动填写超声号和报告姓名。两项与医院档案不一致时会停止查询。</p>
      <p aria-live="polite" className={`mt-5 rounded-lg px-4 py-3 text-sm ${error?"bg-red-50 text-red-700":"bg-slate-50 text-slate-700"}`}>{status}</p>
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

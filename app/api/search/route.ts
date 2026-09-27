import { getChatGPTUser } from "../../chatgpt-auth";
import { pinyin } from "pinyin-pro";

export const dynamic = "force-dynamic";

const API_URL = "http://yy.tianxihosp.com:64681/api/outpatient/getOutpPacsReptMasterVideo";
const SITE_REFERRER = "http://yy.tianxihosp.com/";

type HospitalPayload = { success?: boolean; msg?: string; response?: unknown };

function safeImageUrl(value: unknown): value is string {
  if (typeof value !== "string") return false;
  try {
    const url = new URL(value);
    return url.protocol === "http:" && url.hostname === "yy.tianxihosp.com" &&
      url.pathname.startsWith("/fileServer/pacs/") && /\.(jpg|jpeg|png|gif)$/i.test(url.pathname);
  } catch {
    return false;
  }
}

function normalizeRomanized(value: string) {
  return value.normalize("NFKC").toUpperCase().replace(/[^A-Z0-9]/g, "");
}

function assertImageIdentity(urls: string[], code: string, name: string) {
  const expectedName = normalizeRomanized(pinyin(name, { toneType: "none" }));
  if (!expectedName || !urls.length) throw new Error("无法核对影像档案姓名，已停止查询。请检查超声号和报告单姓名。");
  for (const value of urls) {
    const url = new URL(value);
    const folder = decodeURIComponent(url.pathname.split("/").slice(-2, -1)[0] || "");
    const parts = folder.split("_");
    const folderCode = parts.shift() || "";
    if (/^\d{8}$/.test(parts.at(-1) || "")) parts.pop();
    const folderName = normalizeRomanized(parts.join(" "));
    if (!/^\d{6,20}$/.test(folderCode) || !folderName || (/^\d{8}$/.test(folderCode) && /^\d{6}$/.test(folderName))) continue;
    if (folderCode !== code || folderName !== expectedName) {
      throw new Error("报告单上的超声号或姓名与医院影像档案不一致，已停止查询；没有下载或保存影像。请核对两项信息。");
    }
  }
}

function toBase64(bytes: Uint8Array): string {
  let binary = "";
  for (let offset = 0; offset < bytes.length; offset += 0x8000) {
    binary += String.fromCharCode(...bytes.subarray(offset, offset + 0x8000));
  }
  return btoa(binary);
}

async function fetchImage(url: string, index: number) {
  const imageUrl = new URL(url);
  const response = await fetch(imageUrl.href, {
    headers: { "User-Agent": "Mozilla/5.0", Referer: SITE_REFERRER },
    signal: AbortSignal.timeout(30000),
  });
  if (!response.ok) throw new Error(`影像读取失败（${response.status}）。`);
  const bytes = new Uint8Array(await response.arrayBuffer());
  if (bytes.length > 20 * 1024 * 1024) throw new Error("影像超过 20 MB。");
  const digest = await crypto.subtle.digest("SHA-256", bytes);
  const fingerprint = Array.from(new Uint8Array(digest), n => n.toString(16).padStart(2, "0")).join("");
  return {
    index,
    mime: response.headers.get("content-type")?.split(";")[0] || "image/jpeg",
    ext: imageUrl.pathname.split(".").pop()?.toLowerCase() || "jpg",
    bytes: toBase64(bytes),
    fingerprint,
  };
}

export async function POST(request: Request) {
  const user = await getChatGPTUser();
  if (!user) return Response.json({ error: "请先登录后再使用。" }, { status: 401, headers: { "Cache-Control": "no-store" } });

  let body: { code?: unknown; name?: unknown };
  try { body = await request.json() as { code?: unknown; name?: unknown }; }
  catch { return Response.json({ error: "请求内容无法读取。" }, { status: 400 }); }
  const code = typeof body.code === "string" ? body.code.trim() : "";
  const name = typeof body.name === "string" ? body.name.trim() : "";
  if (!/^\d{6,20}$/.test(code)) return Response.json({ error: "报告照片里的超声号未识别清楚。" }, { status: 400 });
  if (!name || !/[\u3400-\u9fff]/.test(name)) return Response.json({ error: "请填写彩超报告单上的中文姓名后再查询。" }, { status: 400 });

  try {
    const api = new URL(API_URL);
    api.search = new URLSearchParams({ QueryCode: code, QueryType: "4" });
    const response = await fetch(api, {
      headers: { "User-Agent": "Mozilla/5.0", Referer: SITE_REFERRER },
      signal: AbortSignal.timeout(30000),
    });
    if (!response.ok) throw new Error(`医院查询失败（${response.status}）。`);
    const payload = await response.json() as HospitalPayload;
    if (!payload.success) throw new Error(payload.msg || "医院查询没有成功。");

    const all = Array.isArray(payload.response) ? payload.response.filter(safeImageUrl) : [];
    const stills = all.filter(url => /\.(jpg|jpeg|png)$/i.test(new URL(url).pathname));
    const gifs = all.filter(url => /\.gif$/i.test(new URL(url).pathname));
    const urls = (stills.length >= 2 ? [...stills, ...gifs.slice(0, 2)] : [...stills, ...gifs.slice(0, 4)]).slice(0, 28);
    if (!urls.length) return Response.json({ assets: [] }, { headers: { "Cache-Control": "no-store" } });

    assertImageIdentity(urls, code, name);

    const fetched = await Promise.allSettled(urls.map(fetchImage));
    const seen = new Set<string>();
    const assets = fetched.flatMap(result => {
      if (result.status !== "fulfilled" || seen.has(result.value.fingerprint)) return [];
      seen.add(result.value.fingerprint);
      const { fingerprint, ...asset } = result.value;
      return [asset];
    });
    if (!assets.length) throw new Error("医院返回了影像清单，但影像文件无法读取。");
    return Response.json({ assets }, { headers: { "Cache-Control": "no-store" } });
  } catch (error) {
    const message = error instanceof Error ? error.message : "查询失败，请检查网络后重试。";
    return Response.json({ error: message }, { status: 502, headers: { "Cache-Control": "no-store" } });
  }
}

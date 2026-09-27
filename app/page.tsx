import type { Metadata } from "next";
import { requireChatGPTUser, chatGPTSignOutPath } from "./chatgpt-auth";
import AutoFlow from "./auto-flow";

export const dynamic = "force-dynamic";

export const metadata: Metadata = {
  title: "四维正脸自动截图",
  description: "上传彩超报告单照片，自动识别、查询影像并保存两张 3:4 正脸截图。",
};

export default async function Home() {
  await requireChatGPTUser("/");
  return <AutoFlow signOutHref={chatGPTSignOutPath("/")} />;
}

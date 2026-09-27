import type { Metadata } from "next";
import Script from "next/script";
import "./globals.css";

export const metadata: Metadata = {
  title: "四维正脸自动截图",
  description: "上传彩超报告单照片，自动识别、查询影像并保存两张 3:4 正脸截图。",
  other: {
    "codex-preview": "development",
  },
  icons: {
    icon: "/favicon.svg",
    shortcut: "/favicon.svg",
  },
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="zh-CN">
      <body className="antialiased">
        <Script src="/ocr/tesseract.min.js" strategy="beforeInteractive" />
        <Script src="/lib/jszip.min.js" strategy="beforeInteractive" />
        {children}
      </body>
    </html>
  );
}

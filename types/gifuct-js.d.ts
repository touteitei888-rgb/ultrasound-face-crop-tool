declare module "gifuct-js" {
  export function parseGIF(input: ArrayBuffer | Uint8Array): {
    lsd: { width: number; height: number };
  };
  export function decompressFrames(
    gif: unknown,
    buildImagePatches?: boolean,
  ): Array<{
    dims: { left: number; top: number; width: number; height: number };
    patch: Uint8ClampedArray;
    disposalType?: number;
  }>;
}

/** Measuring how big a loaded splat actually is, so the camera can be framed on it. */

import { unpackSplat, type SplatMesh } from "@sparkjsdev/spark";

import type { ObjectBox, Vec3 } from "./frame";

export interface BoundsOptions {
  /**
   * Fraction trimmed from each end of each axis. Splat training leaves a few stray gaussians
   * far from the object; framing on the true extent would zoom out until the object is a dot.
   */
  trim?: number;
  /** Cap on gaussians inspected; above this every Nth is sampled. */
  maxSamples?: number;
}

/**
 * Robust bounds of a loaded splat, in the file's own (re3draw world) coordinates.
 *
 * Returns null if the mesh carries no decoded splats, which is what a failed or empty load
 * looks like.
 */
export function splatBounds(mesh: SplatMesh, { trim = 0.005, maxSamples = 200_000 }: BoundsOptions = {}):
  | ObjectBox
  | null {
  const packed = mesh.packedSplats;
  const count = packed?.numSplats ?? 0;
  if (!packed?.packedArray || count === 0) return null;

  const stride = Math.max(1, Math.ceil(count / maxSamples));
  const sampled = Math.ceil(count / stride);
  const axes = [new Float32Array(sampled), new Float32Array(sampled), new Float32Array(sampled)];
  let n = 0;
  for (let i = 0; i < count; i += stride) {
    const splat = unpackSplat(packed.packedArray, i, packed.splatEncoding);
    axes[0]![n] = splat.center.x;
    axes[1]![n] = splat.center.y;
    axes[2]![n] = splat.center.z;
    n++;
  }

  const ends = axes.map((values) => {
    const sorted = values.subarray(0, n).slice().sort();
    const cut = Math.min(Math.floor(n * trim), Math.floor((n - 1) / 2));
    return [sorted[cut]!, sorted[n - 1 - cut]!] as const;
  });
  return {
    lower: ends.map((e) => e[0]) as unknown as Vec3,
    upper: ends.map((e) => e[1]) as unknown as Vec3,
  };
}

/**
 * What Spark actually does with a re3draw `.ply`, and what the viewer then shows.
 *
 * Both need real WebGL and a real Worker, so they run in headless Chromium against
 * `test/fixtures/axes.ply` - a splat whose geometry is asymmetric on every axis, so a flip, swap
 * or mirror anywhere in the chain cannot cancel out and look fine.
 *
 * The fixture is written by `packages/worker/scripts/make_test_splat.py` in the re3draw world
 * frame: +X red to 0.10 m, +Y green, +Z blue, a white cap on +Z (up) and a yellow cap on +Y
 * (away from the front camera), on a dark disc at z = 0.
 */

import { afterAll, beforeAll, describe, expect, it } from "vitest";

import { openHarness, type Harness } from "./browser";

const FIXTURE = "/test/fixtures/axes.ply";

interface Splat {
  p: [number, number, number];
  c: [number, number, number];
  s: [number, number, number];
  o: number;
}

let harness: Harness;
let decoded: { numSplats: number; splats: Splat[] };

beforeAll(async () => {
  harness = await openHarness();
  decoded = await harness.page.evaluate(
    (url) => (window as never as { decode: (u: string) => Promise<never> }).decode(url),
    FIXTURE,
  );
}, 120_000);

afterAll(async () => harness?.close());

/** Centroid of the splats whose colour is nearest `target`. */
function centroid(target: [number, number, number], tolerance = 0.25) {
  const hit = decoded.splats.filter(
    (s) => Math.hypot(s.c[0] - target[0], s.c[1] - target[1], s.c[2] - target[2]) < tolerance,
  );
  expect(hit.length, `no splats near colour ${target}`).toBeGreaterThan(20);
  const sum = hit.reduce((a, s) => [a[0] + s.p[0], a[1] + s.p[1], a[2] + s.p[2]], [0, 0, 0]);
  return sum.map((v) => v / hit.length) as [number, number, number];
}

describe("Spark decodes re3draw .ply without transforming it", () => {
  it("decodes every gaussian in the file", () => {
    expect(decoded.numSplats).toBe(1020);
  });

  it("keeps +Z as the up axis: the white cap stays on +Z", () => {
    const [x, y, z] = centroid([0.95, 0.95, 0.95]);
    expect(z).toBeGreaterThan(0.1);
    expect(Math.abs(x)).toBeLessThan(0.01);
    expect(Math.abs(y)).toBeLessThan(0.01);
  });

  it("keeps +Y as written: the yellow cap stays on +Y, not on -Y or +Z", () => {
    const [x, y, z] = centroid([0.95, 0.85, 0.1]);
    expect(y).toBeGreaterThan(0.1);
    expect(Math.abs(x)).toBeLessThan(0.01);
    expect(Math.abs(z)).toBeLessThan(0.01);
  });

  it("keeps +X as written, so the scene is not mirrored", () => {
    const red = decoded.splats.filter((s) => s.c[0] > 0.7 && s.c[1] < 0.35 && s.c[2] < 0.35);
    const tip = red.reduce((a, s) => (s.p[0] > a.p[0] ? s : a));
    expect(tip.p[0]).toBeCloseTo(0.1, 2);
    expect(Math.abs(tip.p[1])).toBeLessThan(0.005);
    expect(Math.abs(tip.p[2])).toBeLessThan(0.005);
  });

  it("puts nothing below the mat plane", () => {
    const lowest = Math.min(...decoded.splats.map((s) => s.p[2]));
    expect(lowest).toBeGreaterThan(-0.02);
  });

  it("round-trips scale (metres) and opacity through the packed encoding", () => {
    const s = decoded.splats[0]!;
    for (const axis of s.s) expect(axis).toBeCloseTo(0.004, 3);
    expect(s.o).toBeGreaterThan(0.95);
  });
});

describe("the viewer renders the object the right way up", () => {
  let shot: {
    width: number;
    height: number;
    splats: number;
    box: { lower: number[]; upper: number[] };
    distinctColours: number;
    white: { n: number; x?: number; y?: number };
    red: { n: number; x?: number; y?: number };
    yellow: { n: number };
  };

  beforeAll(async () => {
    shot = await harness.page.evaluate(
      (url) => (window as never as { renderShot: (u: string) => Promise<never> }).renderShot(url),
      FIXTURE,
    );
  }, 120_000);

  it("renders something rather than an empty frame", () => {
    expect(shot.splats).toBe(1020);
    expect(shot.distinctColours).toBeGreaterThan(20);
  });

  it("measures the object at roughly 10 cm, in metres", () => {
    const height = shot.box.upper[2]! - shot.box.lower[2]!;
    expect(height).toBeGreaterThan(0.08);
    expect(height).toBeLessThan(0.16);
  });

  it("puts up (+Z, the white cap) in the upper half of the image", () => {
    expect(shot.white.n).toBeGreaterThan(50);
    expect(shot.white.y!).toBeLessThan(0.45); // y grows downward in image space
  });

  it("puts +X (red) on the right, so the view is not mirrored", () => {
    expect(shot.red.n).toBeGreaterThan(50);
    expect(shot.red.x!).toBeGreaterThan(0.55);
  });

  it("hides the yellow cap, which faces away from the front camera", () => {
    expect(shot.yellow.n).toBeLessThan(shot.white.n);
  });
});

describe("Spark's SPZ generation", () => {
  /**
   * A canary, not a feature test. The worker writes `.spz` with Niantic's reference encoder, which
   * emits version 4 (magic "NGSP", ZSTD streams). No released Spark decodes that - it handles the
   * gzip-wrapped versions 1-3 - so the viewer has to be fed the `.ply` instead. When this test
   * starts failing, Spark has moved to v4 and the viewer can take the 15x smaller file.
   */
  it("still writes gzip-wrapped SPZ, so it cannot read the worker's v4 files", async () => {
    const result = await harness.page.evaluate(
      (url) => (window as never as { spzGeneration: (u: string) => Promise<never> }).spzGeneration(url),
      FIXTURE,
    );
    const { magic, roundTripSplats } = result as unknown as { magic: string; roundTripSplats: number };
    expect(roundTripSplats).toBe(1020); // Spark reads back what Spark writes
    expect(magic.startsWith("1f 8b")).toBe(true); // gzip, not the "NGSP" plaintext header of v4
  });
});

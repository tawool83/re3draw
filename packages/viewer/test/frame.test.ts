/** The re3draw world -> three.js mapping, and the camera framing built on it. */

import * as THREE from "three";
import { describe, expect, it } from "vitest";

import { boxToThree, fitDistance, frameFront, RE3DRAW_TO_THREE, toThree } from "../src/frame";

const close = (v: THREE.Vector3, expected: [number, number, number]) => {
  expect(v.x).toBeCloseTo(expected[0], 6);
  expect(v.y).toBeCloseTo(expected[1], 6);
  expect(v.z).toBeCloseTo(expected[2], 6);
};

describe("world mapping", () => {
  it("sends re3draw up (+Z) to three.js up (+Y)", () => {
    close(toThree([0, 0, 1]), [0, 1, 0]);
  });

  it("leaves +X alone so left and right are not mirrored", () => {
    close(toThree([1, 0, 0]), [1, 0, 0]);
  });

  it("puts the front of the object (-Y) where three.js puts its camera (+Z)", () => {
    close(toThree([0, -1, 0]), [0, 0, 1]);
  });

  it("is a rotation: lengths and handedness survive", () => {
    const m = new THREE.Matrix4().makeRotationFromQuaternion(RE3DRAW_TO_THREE);
    expect(m.determinant()).toBeCloseTo(1, 9);
    const v: [number, number, number] = [0.03, -0.11, 0.07];
    expect(toThree(v).length()).toBeCloseTo(Math.hypot(...v), 9);
  });
});

describe("boxToThree", () => {
  it("keeps the mat plane at y = 0 and the object above it", () => {
    const box = boxToThree({ lower: [-0.12, -0.12, 0], upper: [0.12, 0.12, 0.18] });
    expect(box.min.y).toBeCloseTo(0, 6);
    expect(box.max.y).toBeCloseTo(0.18, 6);
  });

  it("maps the horizontal extents onto x and z", () => {
    const box = boxToThree({ lower: [-0.05, -0.2, 0], upper: [0.05, 0.2, 0.1] });
    expect(box.max.x - box.min.x).toBeCloseTo(0.1, 6);
    expect(box.max.z - box.min.z).toBeCloseTo(0.4, 6);
  });
});

describe("fitDistance", () => {
  const camera = (aspect: number, fov = 50) => new THREE.PerspectiveCamera(fov, aspect, 0.01, 100);
  const box = (size: number) =>
    new THREE.Box3(new THREE.Vector3(-size / 2, 0, -size / 2), new THREE.Vector3(size / 2, size, size / 2));

  it("scales with the object", () => {
    const c = camera(16 / 9);
    expect(fitDistance(box(0.2), c)).toBeCloseTo(2 * fitDistance(box(0.1), c), 6);
  });

  it("backs off further on a narrow viewport, where the limit is horizontal", () => {
    expect(fitDistance(box(0.1), camera(0.5))).toBeGreaterThan(fitDistance(box(0.1), camera(2)));
  });

  it("actually fits: every corner projects inside the frustum", () => {
    const c = camera(0.6);
    const b = box(0.15);
    frameFront(c, b);
    c.updateMatrixWorld();
    const projection = new THREE.Matrix4().multiplyMatrices(c.projectionMatrix, c.matrixWorldInverse);
    for (const corner of cornersOf(b)) {
      const ndc = corner.clone().applyMatrix4(projection);
      expect(Math.abs(ndc.x)).toBeLessThanOrEqual(1);
      expect(Math.abs(ndc.y)).toBeLessThanOrEqual(1);
      expect(Math.abs(ndc.z)).toBeLessThanOrEqual(1);
    }
  });
});

describe("frameFront", () => {
  const box = boxToThree({ lower: [-0.06, -0.06, 0], upper: [0.06, 0.06, 0.12] });

  it("sits in front of the object and above it", () => {
    const camera = new THREE.PerspectiveCamera(50, 1.5, 0.01, 100);
    const target = frameFront(camera, box, { elevationDeg: 20 });
    expect(camera.position.z).toBeGreaterThan(target.z); // in front
    expect(camera.position.y).toBeGreaterThan(target.y); // looking slightly down
    expect(camera.position.x).toBeCloseTo(target.x, 6);
  });

  it("honours the elevation angle", () => {
    const camera = new THREE.PerspectiveCamera(50, 1.5, 0.01, 100);
    for (const elevationDeg of [0, 20, 60]) {
      const target = frameFront(camera, box, { elevationDeg });
      const offset = camera.position.clone().sub(target);
      const measured = THREE.MathUtils.radToDeg(Math.asin(offset.y / offset.length()));
      expect(measured).toBeCloseTo(elevationDeg, 4);
    }
  });

  it("sets near and far around a centimetre-scale scene", () => {
    const camera = new THREE.PerspectiveCamera(50, 1.5, 0.01, 100);
    const target = frameFront(camera, box);
    const distance = camera.position.distanceTo(target);
    expect(camera.near).toBeLessThan(distance);
    expect(camera.far).toBeGreaterThan(distance);
    expect(camera.near).toBeGreaterThan(0);
  });
});

function cornersOf(b: THREE.Box3): THREE.Vector3[] {
  const out: THREE.Vector3[] = [];
  for (let i = 0; i < 8; i++) {
    out.push(new THREE.Vector3(i & 1 ? b.min.x : b.max.x, i & 2 ? b.min.y : b.max.y, i & 4 ? b.min.z : b.max.z));
  }
  return out;
}

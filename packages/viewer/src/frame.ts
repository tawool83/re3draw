/**
 * Coordinate handling between the re3draw world and three.js, and camera framing from it.
 *
 * re3draw world: origin at the mat centre, +X right, +Y toward the mat's top edge, +Z up, metres.
 * The FRONT arrow is on the mat's bottom edge, so the front of the object faces -Y.
 *
 * three.js world: +Y up, and a default camera looks down -Z, i.e. it sits on +Z.
 *
 * Spark loads splat files verbatim - measured against a fixture with known geometry in
 * `test/spark-contract.test.ts` - so this rotation is the viewer's job and nobody else's.
 */

import * as THREE from "three";

export type Vec3 = readonly [number, number, number];

/** Axis-aligned object bounds in the re3draw world frame, metres (`object_box_m` in train.json). */
export interface ObjectBox {
  lower: Vec3;
  upper: Vec3;
}

/**
 * Rotation taking re3draw world axes to three.js world axes: (x, y, z) -> (x, z, -y).
 *
 * +Z (up) becomes +Y (up), and -Y (the front of the object) becomes +Z, which is where three.js
 * puts its camera by default. So an object loaded with this rotation faces the viewer.
 */
export const RE3DRAW_TO_THREE = new THREE.Quaternion().setFromAxisAngle(
  new THREE.Vector3(1, 0, 0),
  -Math.PI / 2,
);

/** A point in the re3draw world frame -> the same point in three.js world space. */
export function toThree(v: Vec3): THREE.Vector3 {
  return new THREE.Vector3(v[0], v[1], v[2]).applyQuaternion(RE3DRAW_TO_THREE);
}

/** An object box in the re3draw world frame -> its three.js axis-aligned bounds. */
export function boxToThree(box: ObjectBox): THREE.Box3 {
  const out = new THREE.Box3();
  for (let corner = 0; corner < 8; corner++) {
    out.expandByPoint(
      toThree([
        (corner & 1 ? box.lower : box.upper)[0],
        (corner & 2 ? box.lower : box.upper)[1],
        (corner & 4 ? box.lower : box.upper)[2],
      ]),
    );
  }
  return out;
}

/**
 * Distance at which `box` just fits the frustum, both directions accounted for.
 *
 * Uses the box's bounding sphere rather than its silhouette, so the fit holds while orbiting
 * instead of clipping as soon as the camera moves off the axis it was computed on.
 */
export function fitDistance(box: THREE.Box3, camera: THREE.PerspectiveCamera, margin = 1.2): number {
  const radius = box.getSize(new THREE.Vector3()).length() / 2;
  if (radius === 0) return margin;
  const vertical = THREE.MathUtils.degToRad(camera.fov);
  const horizontal = 2 * Math.atan(Math.tan(vertical / 2) * camera.aspect);
  return (margin * radius) / Math.sin(Math.min(vertical, horizontal) / 2);
}

export interface FrameOptions {
  /** Height of the camera above the object's horizon, degrees. */
  elevationDeg?: number;
  /** Extra room around the object; 1 exactly fills the frustum. */
  margin?: number;
}

/**
 * Point `camera` at the object from the front, far enough away to see all of it, and return the
 * orbit target. Near and far planes are set from the same distance, because a metric scene is
 * centimetres across and the usual 0.1 near plane would swallow it.
 */
export function frameFront(
  camera: THREE.PerspectiveCamera,
  box: THREE.Box3,
  { elevationDeg = 20, margin = 1.2 }: FrameOptions = {},
): THREE.Vector3 {
  const target = box.getCenter(new THREE.Vector3());
  const distance = fitDistance(box, camera, margin);
  const elevation = THREE.MathUtils.degToRad(elevationDeg);

  camera.position.set(
    target.x,
    target.y + distance * Math.sin(elevation),
    target.z + distance * Math.cos(elevation), // +Z in three.js is the object's front
  );
  camera.near = Math.max(distance / 1000, 1e-4);
  camera.far = distance * 100;
  camera.lookAt(target);
  camera.updateProjectionMatrix();
  return target;
}

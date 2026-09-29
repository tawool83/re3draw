/** The re3draw splat viewer: a three.js scene wired to Spark, framed in real-world units. */

import { SparkRenderer, SplatMesh } from "@sparkjsdev/spark";
import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";

import { splatBounds } from "./bounds";
import { boxToThree, frameFront, RE3DRAW_TO_THREE, type FrameOptions, type ObjectBox } from "./frame";

export interface ViewerOptions extends FrameOptions {
  /** Canvas to render into. A container element gets a canvas appended to fill it. */
  target: HTMLCanvasElement | HTMLElement;
  /** Splat to load immediately (`.ply`, `.spz`, `.splat`, `.ksplat`, `.sog`). */
  url?: string;
  /**
   * Object bounds in metres (`object_box_m` from train.json). Omitted, the viewer measures the
   * loaded splat instead - which is what a capture from someone else's pipeline needs.
   */
  box?: ObjectBox;
  background?: THREE.ColorRepresentation | null;
  /** Draw the mat plane with 1 cm divisions. Only meaningful because the scene is metric. */
  ground?: boolean;
  onProgress?: (fraction: number) => void;
}

export interface LoadedSplat {
  mesh: SplatMesh;
  /** The bounds the camera was framed on, in the re3draw world frame, metres. */
  box: ObjectBox;
  /** Gaussians actually decoded. */
  splats: number;
}

const DEFAULT_BACKGROUND = 0x14161a;

export class SplatViewer {
  readonly canvas: HTMLCanvasElement;
  readonly scene = new THREE.Scene();
  readonly camera: THREE.PerspectiveCamera;
  readonly renderer: THREE.WebGLRenderer;
  readonly controls: OrbitControls;

  private readonly spark: SparkRenderer;
  private readonly options: ViewerOptions;
  private readonly resizeObserver: ResizeObserver;
  private ground: THREE.GridHelper | null = null;
  private mesh: SplatMesh | null = null;
  private box: ObjectBox | null = null;
  private frameRequest = 0;
  private disposed = false;

  constructor(options: ViewerOptions) {
    this.options = options;
    this.canvas = asCanvas(options.target);
    // antialias is off on purpose: Spark's own docs note that MSAA does nothing for splats and
    // costs a lot of fill rate.
    this.renderer = new THREE.WebGLRenderer({ canvas: this.canvas, antialias: false });
    this.renderer.setPixelRatio(Math.min(globalThis.devicePixelRatio ?? 1, 2));

    this.camera = new THREE.PerspectiveCamera(50, 1, 0.01, 100);
    this.scene.background =
      options.background === null ? null : new THREE.Color(options.background ?? DEFAULT_BACKGROUND);

    this.spark = new SparkRenderer({ renderer: this.renderer });
    this.scene.add(this.spark);

    this.controls = new OrbitControls(this.camera, this.canvas);
    this.controls.enableDamping = true;
    this.controls.enablePan = false; // the object is the subject; panning only loses it

    this.resizeObserver = new ResizeObserver(() => this.resize());
    this.resizeObserver.observe(this.canvas);
    this.resize();

    this.renderer.setAnimationLoop(() => this.render());
    if (options.url) void this.load(options.url);
  }

  /**
   * Replace whatever is displayed. Takes a URL, or a `File` so a user can open a splat off their
   * own disk - the only way to look at a training result before the API exists.
   */
  async load(source: string | File): Promise<LoadedSplat> {
    const from =
      typeof source === "string"
        ? { url: source }
        : { fileBytes: await source.arrayBuffer(), fileName: source.name };
    const mesh = new SplatMesh({
      ...from,
      onProgress: (event) =>
        this.options.onProgress?.(event.total > 0 ? event.loaded / event.total : 0),
    });
    // Spark loads files verbatim, so this is where the re3draw world frame becomes three.js's.
    mesh.quaternion.copy(RE3DRAW_TO_THREE);
    try {
      await mesh.initialized;
    } catch (error) {
      mesh.dispose();
      throw describeLoadFailure(error, typeof source === "string" ? source : source.name);
    }
    if (this.disposed) {
      mesh.dispose();
      throw new Error("viewer was disposed while loading");
    }

    this.clearMesh();
    this.mesh = mesh;
    this.scene.add(mesh);

    const box = this.options.box ?? splatBounds(mesh);
    if (!box) throw new Error("the splat decoded to no gaussians");
    this.frame(box);
    if (this.options.ground ?? true) this.addGround(box);

    return { mesh, box, splats: mesh.packedSplats?.numSplats ?? 0 };
  }

  /** Bounds the camera is framed on, in the re3draw world frame; null before anything loads. */
  get loadedBox(): ObjectBox | null {
    return this.box;
  }

  /**
   * Put the camera back in front of the object at a distance that shows all of it. Defaults to
   * the bounds of whatever is loaded, so "show me the front again" needs no arguments.
   */
  frame(box: ObjectBox | null = this.box): void {
    if (!box) return;
    this.box = box;
    this.controls.target.copy(frameFront(this.camera, boxToThree(box), this.options));
    this.controls.update();
  }

  dispose(): void {
    this.disposed = true;
    cancelAnimationFrame(this.frameRequest);
    this.renderer.setAnimationLoop(null);
    this.resizeObserver.disconnect();
    this.clearMesh();
    this.controls.dispose();
    this.spark.dispose();
    this.renderer.dispose();
  }

  private clearMesh(): void {
    if (this.mesh) {
      this.scene.remove(this.mesh);
      this.mesh.dispose();
      this.mesh = null;
    }
    if (this.ground) {
      this.scene.remove(this.ground);
      this.ground.dispose();
      this.ground = null;
    }
    this.box = null;
  }

  /** The mat plane, ruled every centimetre - the scene is metric, so this reads as a real ruler. */
  private addGround(box: ObjectBox): void {
    const width = Math.max(box.upper[0] - box.lower[0], box.upper[1] - box.lower[1]);
    const extent = Math.max(Math.ceil((width * 2) / 0.05) * 0.05, 0.1);
    this.ground = new THREE.GridHelper(extent, Math.round(extent / 0.01), 0x50606f, 0x2a3138);
    this.ground.position.y = 0; // z = 0 in the re3draw world is the surface the object stands on
    this.scene.add(this.ground);
  }

  private resize(): void {
    const width = this.canvas.clientWidth || 1;
    const height = this.canvas.clientHeight || 1;
    this.renderer.setSize(width, height, false);
    this.camera.aspect = width / height;
    this.camera.updateProjectionMatrix();
  }

  private render(): void {
    this.controls.update();
    this.renderer.render(this.scene, this.camera);
  }
}

/**
 * Spark only decodes gzip-wrapped SPZ (versions 1-3). The current reference encoder - which the
 * worker uses - writes version 4, which is ZSTD, so Spark reports a gzip error that says nothing
 * about the real problem. Name it, since the answer is "load the .ply instead".
 */
function describeLoadFailure(error: unknown, name: string): Error {
  const message = error instanceof Error ? error.message : String(error);
  if (/gzip/i.test(message) && /\.spz($|\?)/i.test(name)) {
    return new Error(
      `${name} looks like an SPZ v4 file (ZSTD), which Spark cannot decode yet - it reads ` +
        `versions 1-3 only. Load the .ply from the same training run instead. (${message})`,
    );
  }
  return error instanceof Error ? error : new Error(message);
}

function asCanvas(target: HTMLCanvasElement | HTMLElement): HTMLCanvasElement {
  if (target instanceof HTMLCanvasElement) return target;
  const canvas = target.ownerDocument.createElement("canvas");
  canvas.style.width = "100%";
  canvas.style.height = "100%";
  canvas.style.display = "block";
  target.appendChild(canvas);
  return canvas;
}

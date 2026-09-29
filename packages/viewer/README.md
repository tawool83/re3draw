# @re3draw/viewer

Three.js + [Spark](https://sparkjs.dev) viewer for re3draw splats, shared by the web SDK (M3) and
the Flutter WebView (M4).

```bash
npm install
npm run dev     # demo page: loads the reference fixture, or open your own splat.ply
npm test        # pure maths in node, plus a real render in headless Chromium
npm run build   # dist/re3draw-viewer.js (ES) + dist/re3draw-viewer.embed.js (IIFE) + types
```

```ts
import { SplatViewer } from "@re3draw/viewer";

const viewer = new SplatViewer({ target: document.getElementById("stage")! });
const { box, splats } = await viewer.load("/captures/mug/splat.spz");
```

Formats come from Spark: `.ply`, `.spz`, `.splat`, `.ksplat`, `.sog`. `load()` also takes a `File`,
so a user can open a training result straight off disk.

> **Feed it the `.ply`, not the `.spz`.** The worker writes `.spz` with Niantic's reference encoder,
> which emits **version 4** (magic `NGSP`, ZSTD streams). No released Spark decodes that — it reads
> the gzip-wrapped versions 1–3 — so a v4 file fails with `Invalid gzip header`. `load()` catches
> that and says so. It costs 15x in transfer size (21 MB vs 1.4 MB on the reference capture), which
> M3 has to solve: wait for Spark, emit an older SPZ as well, or move to `.sog`.
> `test/spark-contract.test.ts` pins Spark's current SPZ generation so the day it changes is loud.

## The coordinate frame, which is the whole problem

re3draw's world is the printed mat: origin at its centre, **+X** right, **+Y** toward the top edge,
**+Z** up, metres — and the FRONT arrow on the bottom edge means the object faces **−Y**.
Three.js is Y-up and puts its camera on **+Z**.

**Spark loads splat files verbatim.** It applies no flip of its own, which is not in its docs and is
therefore measured instead: `test/spark-contract.test.ts` decodes a fixture with known geometry and
asserts each axis comes back where it was written. So converting the frame is this package's job:

```
(x, y, z)  ->  (x, z, -y)        RE3DRAW_TO_THREE, a -90 deg rotation about X
```

Up becomes up, and the object's front ends up facing three.js's default camera, so a freshly loaded
capture faces the viewer instead of showing its back.

## Metric scale

Because the mat fixes real-world scale, the scene is in metres and the viewer uses it:

- the camera is framed from the object's true size, not a guess;
- near and far planes are set from that distance — the usual `0.1` near plane would swallow a 12 cm
  object entirely;
- the ground grid is ruled every centimetre, so the object is visibly *some number of centimetres*;
- `load()` returns the bounds it measured, which is where the demo's `13.2 x 16.3 x 13.2 cm` reading
  comes from.

Bounds come from `object_box_m` in `train.json` when you pass `box`, and are otherwise measured from
the splat itself, trimming the outer 0.5 % per axis so a few stray gaussians cannot zoom the object
down to a dot.

## Tests

| File | Needs | Covers |
| --- | --- | --- |
| `test/frame.test.ts` | node | the world mapping, and that every corner of the object really projects inside the frustum |
| `test/spark-contract.test.ts` | headless Chromium (WebGL) | what Spark does to coordinates, that a rendered frame has up at the top and +X on the right, and which SPZ generation Spark speaks |

The fixture `test/fixtures/axes.ply` is written by
`packages/worker/scripts/make_test_splat.py`. It is deliberately asymmetric on all three axes —
red +X, a white cap on +Z, a yellow cap on +Y — so no flip, swap or mirror can cancel out and still
look correct. Regenerate it with:

```bash
cd ../worker && .venv/bin/python scripts/make_test_splat.py ../viewer/test/fixtures/axes.ply
```

Chromium comes from Playwright and renders through SwiftShader, so the render test needs no GPU.

## Two builds

| Output | For | Bundled |
| --- | --- | --- |
| `dist/re3draw-viewer.js` | the web SDK | nothing — `three` and Spark stay external so the host ships one copy |
| `dist/re3draw-viewer.embed.js` | the Flutter WebView | everything, since it loads from local assets with no module loader and no CDN |

## Not done yet

- `.spz` loading, until Spark reads version 4 (see above)
- `.glb` mesh display — the worker does not produce meshes yet
- touch-gesture tuning and a mobile pass
- progressive / level-of-detail loading for large splats (Spark supports it; not wired up)

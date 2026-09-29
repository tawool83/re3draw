import { resolve } from "node:path";
import { defineConfig } from "vite";

// Two consumers, two builds:
//   default  - ES module for the web SDK. three and Spark stay external so the host application
//              bundles one copy of each (Spark alone is ~2.5 MB)
//   embed    - one self-contained IIFE for the Flutter WebView, which has no module loader and,
//              being loaded from local assets, no CDN to fetch dependencies from
export default defineConfig(({ mode }) => {
  const embed = mode === "embed";
  return {
    build: {
      emptyOutDir: !embed,
      lib: {
        entry: resolve(__dirname, "src/index.ts"),
        name: "Re3drawViewer",
        formats: embed ? ["iife"] : ["es"],
        fileName: () => (embed ? "re3draw-viewer.embed.js" : "re3draw-viewer.js"),
      },
      rollupOptions: embed ? {} : { external: ["three", /^three\//, "@sparkjsdev/spark"] },
    },
    test: {
      environment: "node",
      include: ["test/**/*.test.ts"],
      testTimeout: 60_000,
    },
  };
});

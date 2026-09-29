/** Shared harness: a Vite dev server plus a headless Chromium, for tests that need real WebGL. */

import { chromium, type Browser, type Page } from "playwright";
import { createServer, type ViteDevServer } from "vite";

export interface Harness {
  page: Page;
  close: () => Promise<void>;
}

export async function openHarness(): Promise<Harness> {
  const server: ViteDevServer = await createServer({
    configFile: false,
    root: process.cwd(),
    server: { port: 0 },
    // three's addons are deep imports; letting Vite pre-bundle them keeps the page to one request
    optimizeDeps: { include: ["three", "@sparkjsdev/spark"] },
  });
  await server.listen();
  const port = server.config.server.port ?? server.httpServer?.address();

  const browser: Browser = await chromium.launch();
  const page = await browser.newPage({ viewport: { width: 800, height: 600 } });
  const errors: string[] = [];
  page.on("pageerror", (e) => errors.push(e.message));
  page.on("console", (m) => {
    if (m.type() === "error") errors.push(m.text());
  });

  await page.goto(`http://localhost:${port}/test/harness.html`);
  try {
    await page.waitForFunction(() => (window as never as { READY?: boolean }).READY === true, null, {
      timeout: 30_000,
    });
  } catch (e) {
    throw new Error(`harness failed to start: ${errors.join(" | ") || String(e)}`);
  }

  return {
    page,
    close: async () => {
      await browser.close();
      await server.close();
      if (errors.length) console.warn("browser errors:", errors.join(" | "));
    },
  };
}

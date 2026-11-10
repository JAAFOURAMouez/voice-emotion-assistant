import { svelte } from "@sveltejs/vite-plugin-svelte";
import { defineConfig } from "vite";

// In development the page is served by Vite (localhost:5173) and the API by
// `python -m voxemo.api` (localhost:8000): /api is proxied, WebSocket included.
export default defineConfig({
  plugins: [svelte()],
  server: {
    proxy: {
      "/api": { target: "http://127.0.0.1:8000", ws: true },
    },
  },
  build: { outDir: "dist", assetsInlineLimit: 0 },
  test: { environment: "node", include: ["src/**/*.test.ts"] },
});

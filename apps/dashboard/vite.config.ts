import react from "@vitejs/plugin-react";
import { defineConfig } from "vitest/config";

// In development the dashboard talks to the API through this proxy, so the browser only
// ever sees one origin. In Docker, nginx does the same job (see infrastructure/docker).
export default defineConfig({
  plugins: [react()],
  test: { environment: "jsdom", include: ["src/**/*.test.{ts,tsx}"], restoreMocks: true, setupFiles: ["src/test/setup.ts"] },
  server: {
    port: 5173,
    proxy: { "/api": { target: process.env.AEGIS_API_URL ?? "http://localhost:8000", rewrite: (p) => p.replace(/^\/api/, "") } },
  },
});

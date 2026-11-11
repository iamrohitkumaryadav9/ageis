import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

// In development the dashboard talks to the API through this proxy, so the browser only
// ever sees one origin. In Docker, nginx does the same job (see infrastructure/docker).
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: { "/api": { target: process.env.AEGIS_API_URL ?? "http://localhost:8000", rewrite: (p) => p.replace(/^\/api/, "") } },
  },
});

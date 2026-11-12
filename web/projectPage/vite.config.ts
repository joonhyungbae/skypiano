import { defineConfig } from "vite";
import react from "@vitejs/plugin-react-swc";
import path from "path";

// https://vitejs.dev/config/
export default defineConfig({
  // Served from the GitHub Pages project path of the skypiano repository;
  // the explorer build sits one level below this, at /skypiano/explorer/.
  base: "/skypiano/",
  server: {
    host: "::",
    port: 3141,
    open: "/skypiano/",
    proxy: {
      // In production the explorer is a static build nested under this app.
      // In dev it runs as its own server, so forward its mount point there and
      // the "Open explorer" links work from one URL with HMR on both sides.
      "/skypiano/explorer": {
        target: "http://localhost:9999",
        changeOrigin: true,
        ws: true,
      },
    },
  },
  plugins: [react()],
  resolve: {
    alias: {
      "@": path.resolve(__dirname, "./src"),
    },
  },
});

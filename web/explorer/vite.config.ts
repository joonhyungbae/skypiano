import { fileURLToPath, URL } from 'node:url'

import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// Two builds from one source:
//   npm run build           the public explorer (curated samples), mounted under the project page
//   npm run build:dataset   the standalone full-dataset explorer served by serve_dataset.py
//                           (base '/', no project-page or sample navigation; .env.dataset)
export default defineConfig(({ mode }) => ({
  base: mode === 'dataset' ? '/' : '/skypiano/explorer/',
  build: { outDir: mode === 'dataset' ? 'dist-dataset' : 'dist' },
  // the sample media in public/ belong to the public site; serve_dataset.py serves keyboard.png itself
  publicDir: mode === 'dataset' ? false : 'public',
  plugins: [react()],
  resolve: {
    alias: {
      '@': fileURLToPath(new URL('./src', import.meta.url))
    }
  },
  server: {
    port: 9999,
    // Bind to all interfaces so the dev server is reachable through the
    // Tailscale IP — Tailscale intercepts plain localhost on this machine
    // and Chrome ends up with ERR_EMPTY_RESPONSE on http://localhost:9999.
    host: true,
    hmr: {
      protocol: 'ws',
      clientPort: 9999,
    },
  },
}))

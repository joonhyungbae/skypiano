import { fileURLToPath, URL } from 'node:url'

import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vitejs.dev/config/
export default defineConfig({
  // Mounted under the project page, which owns the /skypiano/ root.
  base: '/skypiano/explorer/',
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
})

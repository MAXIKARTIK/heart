import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// During `npm run dev`, /api is proxied to the FastAPI backend so the browser
// talks to a single origin (avoids CORS in local dev). In production the app is
// served behind nginx which proxies /api to the api container.
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      '/api': {
        target: 'http://localhost:8000',
        changeOrigin: true,
      },
    },
  },
})

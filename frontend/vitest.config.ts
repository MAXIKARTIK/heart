import { defineConfig } from 'vitest/config'
import react from '@vitejs/plugin-react'

// Vitest configuration for the frontend test suite (task 17.1 test harness).
//
// - `react()` gives the same JSX / React 18 automatic-runtime transform the app
//   build uses, so components render under test exactly as they do in the app.
// - `environment: 'jsdom'` supplies a browser-like DOM so React Testing Library
//   can mount and query components.
// - `setupFiles` runs once per test worker to register the
//   `@testing-library/jest-dom` matchers and reset the DOM between tests.
// - `globals: true` exposes `describe`/`it`/`expect` without importing them in
//   every test file (typed via the reference in `src/test/setup.ts`).
//
// Vitest prefers this file over `vite.config.ts` when both are present, so the
// dev server / production build config stays untouched.
export default defineConfig({
  plugins: [react()],
  test: {
    globals: true,
    environment: 'jsdom',
    setupFiles: ['./src/test/setup.ts'],
    include: ['src/**/*.{test,spec}.{ts,tsx}'],
  },
})

/// <reference types="vitest/globals" />
import { afterEach } from 'vitest'
import { cleanup } from '@testing-library/react'

// Registers the jest-dom matchers (e.g. toBeInTheDocument, toHaveValue) on
// Vitest's `expect` and augments the matcher types.
import '@testing-library/jest-dom/vitest'

// Unmount anything React Testing Library rendered so each test starts with a
// clean DOM, regardless of whether auto-cleanup is active.
afterEach(() => {
  cleanup()
})

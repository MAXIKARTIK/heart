import type { ClinicalFeatures, ModelInfo, PredictionResponse } from './types'

// In dev this is '' (Vite proxies /api to the backend). In production set
// VITE_API_URL to the API origin if the frontend and API are on different hosts.
const BASE = (import.meta.env.VITE_API_URL ?? '') + '/api/v1'

async function handle<T>(res: Response): Promise<T> {
  if (!res.ok) {
    let detail = res.statusText
    try {
      const body = await res.json()
      detail = typeof body.detail === 'string' ? body.detail : JSON.stringify(body.detail)
    } catch {
      /* non-JSON error body */
    }
    throw new Error(`Request failed (${res.status}): ${detail}`)
  }
  return (await res.json()) as T
}

export async function predict(features: ClinicalFeatures): Promise<PredictionResponse> {
  const res = await fetch(`${BASE}/predict`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(features),
  })
  return handle<PredictionResponse>(res)
}

export async function getModelInfo(): Promise<ModelInfo> {
  const res = await fetch(`${BASE}/health/model`)
  return handle<ModelInfo>(res)
}

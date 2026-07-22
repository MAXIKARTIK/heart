import { useState } from 'react'
import { useMutation } from '@tanstack/react-query'
import { predict } from '../api'
import {
  CATEGORICAL_FIELDS,
  CATEGORY_VALUES,
  DEFAULT_FEATURES,
  NUMERIC_FIELDS,
} from '../constants'
import type { ClinicalFeatures, PredictionResponse } from '../types'
import ResultCard from './ResultCard'

export default function PredictionForm() {
  const [form, setForm] = useState<ClinicalFeatures>(DEFAULT_FEATURES)

  const mutation = useMutation<PredictionResponse, Error, ClinicalFeatures>({
    mutationFn: predict,
  })

  const setNumeric = (name: keyof ClinicalFeatures, value: string) =>
    setForm((prev) => ({ ...prev, [name]: value === '' ? 0 : Number(value) }))

  const setCategory = (name: keyof ClinicalFeatures, value: string) =>
    setForm((prev) => ({ ...prev, [name]: value }))

  const onSubmit = (e: React.FormEvent<HTMLFormElement>) => {
    e.preventDefault()
    mutation.mutate(form)
  }

  return (
    <form className="form" onSubmit={onSubmit}>
      <div className="grid">
        {NUMERIC_FIELDS.map((f) => (
          <label key={f.name} className="field">
            <span>{f.label}</span>
            <input
              type="number"
              min={f.min}
              max={f.max}
              step={f.step}
              value={form[f.name] as number}
              onChange={(e) => setNumeric(f.name, e.target.value)}
              required
            />
          </label>
        ))}

        {CATEGORICAL_FIELDS.map((f) => (
          <label key={f.name} className="field">
            <span>{f.label}</span>
            <select
              value={form[f.name] as string}
              onChange={(e) => setCategory(f.name, e.target.value)}
            >
              {CATEGORY_VALUES[f.name].map((opt) => (
                <option key={opt} value={opt}>
                  {opt}
                </option>
              ))}
            </select>
          </label>
        ))}
      </div>

      <div className="actions">
        <button type="submit" disabled={mutation.isPending}>
          {mutation.isPending ? 'Scoring…' : 'Predict risk'}
        </button>
        <button
          type="button"
          className="secondary"
          onClick={() => {
            setForm(DEFAULT_FEATURES)
            mutation.reset()
          }}
        >
          Reset
        </button>
      </div>

      {mutation.isError && <div className="error">{mutation.error.message}</div>}
      {mutation.isSuccess && <ResultCard result={mutation.data} />}
    </form>
  )
}

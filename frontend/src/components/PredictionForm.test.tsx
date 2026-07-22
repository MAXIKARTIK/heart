import { render, screen, fireEvent, within } from '@testing-library/react'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import type { ReactElement } from 'react'
import { beforeEach, describe, expect, it, vi } from 'vitest'

import PredictionForm from './PredictionForm'
import {
  CATEGORICAL_FIELDS,
  CATEGORY_VALUES,
  DEFAULT_FEATURES,
  NUMERIC_FIELDS,
} from '../constants'
import type { PredictionResponse } from '../types'
import { predict } from '../api'

// The form submits through the real `predict` API helper via TanStack Query's
// useMutation. Mock the api module so the component's `import { predict }` picks
// up a controllable stub instead of hitting the network. (Requirements 17.3, 17.4)
vi.mock('../api', () => ({
  predict: vi.fn(),
}))

const mockPredict = vi.mocked(predict)

// PredictionForm relies on a QueryClientProvider for useMutation; retries are
// disabled so the error case surfaces immediately without backoff.
function renderWithClient(ui: ReactElement) {
  const queryClient = new QueryClient({
    defaultOptions: {
      queries: { retry: false },
      mutations: { retry: false },
    },
  })
  return render(<QueryClientProvider client={queryClient}>{ui}</QueryClientProvider>)
}

beforeEach(() => {
  mockPredict.mockReset()
})

describe('PredictionForm', () => {
  // Requirement 17.2: numeric inputs are bounded to each field's allowed range.
  it('renders each numeric field as a number input bounded to its min/max/step', () => {
    renderWithClient(<PredictionForm />)

    for (const field of NUMERIC_FIELDS) {
      const input = screen.getByLabelText(field.label)
      expect(input).toHaveAttribute('type', 'number')
      expect(input).toHaveAttribute('min', String(field.min))
      expect(input).toHaveAttribute('max', String(field.max))
      expect(input).toHaveAttribute('step', String(field.step))
    }
  })

  // Requirement 17.2: categorical inputs are limited to each field's allowed values.
  it('renders each categorical field as a dropdown limited to its allowed values', () => {
    renderWithClient(<PredictionForm />)

    for (const field of CATEGORICAL_FIELDS) {
      const select = screen.getByLabelText(field.label)
      const options = within(select).getAllByRole('option')
      const allowed = CATEGORY_VALUES[field.name]

      // The dropdown offers exactly the allowed values (labels and submitted
      // values), and nothing outside the allowed set.
      expect(options.map((o) => o.textContent)).toEqual(allowed)
      expect(options.map((o) => (o as HTMLOptionElement).value)).toEqual(allowed)
    }
  })

  // Requirement 17.3: submitting sends the entered features and displays the
  // returned risk label and probability (model internals are not shown).
  it('sends the entered features and displays the returned risk label and probability', async () => {
    const response: PredictionResponse = {
      id: 42,
      prediction: 1,
      risk_label: 'At risk',
      probability: 0.734,
      threshold: 0.42,
      model_name: 'DecisionTree',
      trained_at: '2024-01-01T00:00:00Z',
    }
    mockPredict.mockResolvedValue(response)

    renderWithClient(<PredictionForm />)

    // Enter values into a numeric field and a categorical field.
    fireEvent.change(screen.getByLabelText('BMI'), { target: { value: '33.3' } })
    fireEvent.change(screen.getByLabelText('Sex'), { target: { value: 'Female' } })

    fireEvent.click(screen.getByRole('button', { name: /predict risk/i }))

    // The returned risk label is shown once the mutation resolves.
    expect(await screen.findByText(response.risk_label)).toBeInTheDocument()

    // Probability is rendered exactly as ResultCard formats it.
    const pct = (response.probability * 100).toFixed(1)
    expect(screen.getByText(`${pct}%`)).toBeInTheDocument()

    // The submitted payload reflects the entered Clinical_Features. (React Query
    // passes a mutation-context object as a second arg, so assert on the first.)
    expect(mockPredict).toHaveBeenCalledTimes(1)
    expect(mockPredict.mock.calls[0][0]).toEqual({
      ...DEFAULT_FEATURES,
      BMI: 33.3,
      Sex: 'Female',
    })
  })

  // Requirement 17.4: a failed prediction request shows a descriptive error banner.
  it('shows an error banner when the prediction request fails', async () => {
    const message = 'Request failed (503): Model not loaded'
    mockPredict.mockRejectedValue(new Error(message))

    renderWithClient(<PredictionForm />)

    fireEvent.click(screen.getByRole('button', { name: /predict risk/i }))

    const banner = await screen.findByText(message)
    expect(banner).toHaveClass('error')
    // No result card is rendered on failure.
    expect(screen.queryByText(/estimated probability of heart disease/i)).not.toBeInTheDocument()
  })
})

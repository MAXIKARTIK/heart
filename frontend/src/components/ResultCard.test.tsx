import { render, screen } from '@testing-library/react'
import ResultCard from './ResultCard'
import type { PredictionResponse } from '../types'

// ResultCard is a pure presentational component: given a PredictionResponse it
// renders the risk label, the estimated probability (as a one-decimal
// percentage), and a fixed not-a-medical-device disclaimer. Model internals
// (name, decision threshold) are intentionally NOT shown in the patient-facing
// UI. These tests lock in that behavior against Requirements 17.3 (display risk
// label and probability) and 17.5 (always display the demonstration-only
// disclaimer).

function makeResult(
  overrides: Partial<PredictionResponse> = {},
): PredictionResponse {
  return {
    id: 42,
    prediction: 1,
    risk_label: 'At risk',
    probability: 0.732,
    threshold: 0.5,
    model_name: 'DecisionTree',
    trained_at: '2024-01-01T00:00:00Z',
    ...overrides,
  }
}

describe('ResultCard', () => {
  it('renders the risk label from the prediction response (Req 17.3)', () => {
    render(<ResultCard result={makeResult({ risk_label: 'At risk' })} />)

    expect(screen.getByText('At risk')).toBeInTheDocument()
  })

  it('renders the probability as a one-decimal percentage (Req 17.3)', () => {
    // 0.732 -> "73.2%"
    render(<ResultCard result={makeResult({ probability: 0.732 })} />)

    expect(screen.getByText('73.2%')).toBeInTheDocument()
  })

  it('always shows the not-a-medical-device disclaimer (Req 17.5)', () => {
    render(<ResultCard result={makeResult()} />)

    expect(screen.getByText(/for demonstration only/i)).toBeInTheDocument()
    expect(
      screen.getByText(
        /not a medical device and not a substitute for professional diagnosis/i,
      ),
    ).toBeInTheDocument()
  })

  it('renders label, probability, and disclaimer for a low-risk result too (Reqs 17.3, 17.5)', () => {
    // The disclaimer must appear regardless of the outcome, so exercise the
    // negative-prediction branch as well.
    render(
      <ResultCard
        result={makeResult({
          prediction: 0,
          risk_label: 'Low risk',
          probability: 0.031,
        })}
      />,
    )

    expect(screen.getByText('Low risk')).toBeInTheDocument()
    expect(screen.getByText('3.1%')).toBeInTheDocument()
    expect(
      screen.getByText(/not a substitute for professional diagnosis/i),
    ).toBeInTheDocument()
  })
})

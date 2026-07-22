import type { PredictionResponse } from '../types'

export default function ResultCard({ result }: { result: PredictionResponse }) {
  const atRisk = result.prediction === 1
  const pct = (result.probability * 100).toFixed(1)

  return (
    <div className={`result-card ${atRisk ? 'risk' : 'safe'}`}>
      <div className="result-label">{result.risk_label}</div>
      <div className="result-prob">
        <span className="result-prob-value">{pct}%</span>
        <span className="result-prob-caption">estimated probability of heart disease</span>
      </div>
      <div className="result-gauge">
        <div className="result-gauge-fill" style={{ width: `${pct}%` }} />
      </div>
      <p className="disclaimer">
        For demonstration only. Not a medical device and not a substitute for
        professional diagnosis.
      </p>
    </div>
  )
}

import PredictionForm from './components/PredictionForm'

export default function App() {
  return (
    <div className="app">
      <header className="header">
        <h1>Heart Disease Risk Predictor</h1>
        <p>
          Enter the patient's clinical details below to estimate their risk of
          heart disease.
        </p>
      </header>

      <main>
        <PredictionForm />
      </main>

      <footer className="footer">
        For educational use only — not a medical device.
      </footer>
    </div>
  )
}

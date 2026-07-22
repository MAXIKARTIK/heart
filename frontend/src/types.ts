export interface ClinicalFeatures {
  BMI: number
  Smoking: string
  AlcoholDrinking: string
  Stroke: string
  PhysicalHealth: number
  MentalHealth: number
  DiffWalking: string
  Sex: string
  AgeCategory: string
  Race: string
  Diabetic: string
  PhysicalActivity: string
  GenHealth: string
  SleepTime: number
  Asthma: string
  KidneyDisease: string
  SkinCancer: string
}

export interface PredictionResponse {
  id: number | null
  prediction: number
  risk_label: string
  probability: number
  threshold: number
  model_name: string
  trained_at: string
}

export interface ModelInfo {
  model_name: string
  threshold: number
  trained_at: string
  sklearn_version: string
  n_train: number
  positive_rate: number
  metrics: Record<string, number>
}

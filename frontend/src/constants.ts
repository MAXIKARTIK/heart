import type { ClinicalFeatures } from './types'

// Mirrors heart_ml.config.CATEGORY_VALUES (the exact values in the CDC dataset).
export const CATEGORY_VALUES: Record<string, string[]> = {
  Smoking: ['No', 'Yes'],
  AlcoholDrinking: ['No', 'Yes'],
  Stroke: ['No', 'Yes'],
  DiffWalking: ['No', 'Yes'],
  Sex: ['Female', 'Male'],
  AgeCategory: [
    '18-24', '25-29', '30-34', '35-39', '40-44', '45-49', '50-54',
    '55-59', '60-64', '65-69', '70-74', '75-79', '80 or older',
  ],
  Race: ['White', 'Black', 'Asian', 'American Indian/Alaskan Native', 'Hispanic', 'Other'],
  Diabetic: ['No', 'No, borderline diabetes', 'Yes', 'Yes (during pregnancy)'],
  PhysicalActivity: ['No', 'Yes'],
  GenHealth: ['Poor', 'Fair', 'Good', 'Very good', 'Excellent'],
  Asthma: ['No', 'Yes'],
  KidneyDisease: ['No', 'Yes'],
  SkinCancer: ['No', 'Yes'],
}

export interface NumericField {
  name: keyof ClinicalFeatures
  label: string
  min: number
  max: number
  step: number
}

export const NUMERIC_FIELDS: NumericField[] = [
  { name: 'BMI', label: 'BMI', min: 10, max: 100, step: 0.1 },
  { name: 'PhysicalHealth', label: 'Physical unwell days (last 30)', min: 0, max: 30, step: 1 },
  { name: 'MentalHealth', label: 'Mental unwell days (last 30)', min: 0, max: 30, step: 1 },
  { name: 'SleepTime', label: 'Sleep hours (per 24h)', min: 0, max: 24, step: 0.5 },
]

// Human-friendly labels for the categorical dropdowns.
export const CATEGORICAL_FIELDS: { name: keyof ClinicalFeatures; label: string }[] = [
  { name: 'Sex', label: 'Sex' },
  { name: 'AgeCategory', label: 'Age category' },
  { name: 'Race', label: 'Race / ethnicity' },
  { name: 'GenHealth', label: 'General health' },
  { name: 'Smoking', label: 'Smoker' },
  { name: 'AlcoholDrinking', label: 'Heavy alcohol use' },
  { name: 'PhysicalActivity', label: 'Physically active' },
  { name: 'DiffWalking', label: 'Difficulty walking' },
  { name: 'Stroke', label: 'History of stroke' },
  { name: 'Diabetic', label: 'Diabetic' },
  { name: 'Asthma', label: 'Asthma' },
  { name: 'KidneyDisease', label: 'Kidney disease' },
  { name: 'SkinCancer', label: 'Skin cancer' },
]

export const DEFAULT_FEATURES: ClinicalFeatures = {
  BMI: 28.5,
  Smoking: 'No',
  AlcoholDrinking: 'No',
  Stroke: 'No',
  PhysicalHealth: 2,
  MentalHealth: 2,
  DiffWalking: 'No',
  Sex: 'Male',
  AgeCategory: '55-59',
  Race: 'White',
  Diabetic: 'No',
  PhysicalActivity: 'Yes',
  GenHealth: 'Good',
  SleepTime: 7,
  Asthma: 'No',
  KidneyDisease: 'No',
  SkinCancer: 'No',
}

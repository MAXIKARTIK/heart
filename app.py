from flask import Flask, render_template, request
import pandas as pd
import joblib
from scipy.stats import boxcox

app = Flask(__name__)

model = joblib.load("best_svm.pkl")
lambdas = joblib.load("boxcox_lambdas.pkl")

continuous_features = [
    'age',
    'trestbps',
    'chol',
    'thalach',
    'oldpeak'
]

feature_order = [
    'age',
    'sex',
    'trestbps',
    'chol',
    'fbs',
    'thalach',
    'exang',
    'oldpeak',
    'slope',
    'ca',
    'cp_1',
    'cp_2',
    'cp_3',
    'restecg_1',
    'restecg_2',
    'thal_1',
    'thal_2',
    'thal_3'
]

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():

    age = float(request.form['age'])
    sex = int(request.form['sex'])
    cp = int(request.form['cp'])
    trestbps = float(request.form['trestbps'])
    chol = float(request.form['chol'])
    fbs = int(request.form['fbs'])
    restecg = int(request.form['restecg'])
    thalach = float(request.form['thalach'])
    exang = int(request.form['exang'])
    oldpeak = float(request.form['oldpeak'])
    slope = int(request.form['slope'])
    ca = int(request.form['ca'])
    thal = int(request.form['thal'])

    df = pd.DataFrame({
        'age':[age],
        'sex':[sex],
        'trestbps':[trestbps],
        'chol':[chol],
        'fbs':[fbs],
        'thalach':[thalach],
        'exang':[exang],
        'oldpeak':[oldpeak],
        'slope':[slope],
        'ca':[ca]
    })

    # CP Dummies
    df['cp_1'] = 1 if cp == 1 else 0
    df['cp_2'] = 1 if cp == 2 else 0
    df['cp_3'] = 1 if cp == 3 else 0

    # Rest ECG Dummies
    df['restecg_1'] = 1 if restecg == 1 else 0
    df['restecg_2'] = 1 if restecg == 2 else 0

    # Thal Dummies
    df['thal_1'] = 1 if thal == 1 else 0
    df['thal_2'] = 1 if thal == 2 else 0
    df['thal_3'] = 1 if thal == 3 else 0

    # Box-Cox preprocessing
    df['oldpeak'] += 0.001

    for col in continuous_features:
        df[col] = boxcox(df[col], lmbda=lambdas[col])

    df = df[feature_order]

    probs = model.predict_proba(df)[0]

    prediction = model.predict(df)[0]

    probability = probs[prediction]
    
    if prediction == 1:
        result = "Heart Disease Detected"
    else:
        result = "No Heart Disease"

    return render_template(
        'index.html',
        prediction=result,
        probability=round(probability * 100, 2)
    )

if __name__ == "__main__":
    app.run(debug=True)
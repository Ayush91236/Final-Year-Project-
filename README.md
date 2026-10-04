# Diabetes Risk Predictor

A web app that estimates a person's diabetes risk level (Low / Moderate / High) from basic health and lifestyle information. Built as a final year project (B.Tech CSE, AI/ML).

> **Disclaimer:** This tool is for awareness only and is not a medical diagnosis.

## How it works

1. The user fills in a short form (age, sex, BMI, general health, blood pressure, cholesterol, physical activity, smoking, heart disease, difficulty walking).
2. The app converts the answers into the 21 features the model expects. Features not asked in the form use common default values from the dataset.
3. A Random Forest model predicts a risk score, which is shown as a risk level.

## Dataset

- BRFSS 2015 Diabetes Health Indicators (253,680 survey records).
- After removing 23,899 duplicate rows: 229,781 records (82.7% healthy, 17.3% at risk).
- Target converted to binary: `Diabetes_binary` (0 = healthy, 1 = at risk / diabetic).

## Model

| Model | ROC-AUC | Recall (at risk) |
|---|---|---|
| Logistic Regression (balanced) | 0.805 | 75% |
| **Random Forest (final)** | **0.810** | **75%** |

Random Forest was chosen for its higher ROC-AUC and 75% recall on at-risk patients (class weights were balanced to handle the imbalanced data). Top risk factors: high blood pressure, general health, BMI, age and high cholesterol.

Risk levels are based on the model score: Low (below 0.30), Moderate (0.30 to 0.60), High (0.60 and above).

## Tech stack

Python, Flask, scikit-learn, pandas, HTML/CSS. Dashboard built in Power BI.

## Run locally

```
git clone https://github.com/Ayush91236/Final-Year-Project-.git
cd Final-Year-Project-
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python app.py
```

Then open http://127.0.0.1:5000 in your browser. Python 3.12 is recommended (scikit-learn 1.6.1 is required to load the saved model).

## Limitations

- Trained only on self-reported survey data (BRFSS 2015).
- Does not use family history, diet details, genetics or lab values such as glucose or HbA1c.
- Not a diagnostic tool.

## Future work

- Combine with a richer dataset that includes family history and lab values.
- Add more detailed explanations of each prediction.
  

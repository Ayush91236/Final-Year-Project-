from flask import Flask, render_template, request
import joblib
import pandas as pd

app = Flask(__name__)

# Model aur features ki list, app start hote hi ek baar load hoti hai
model = joblib.load('diabetes_rf_model.pkl')
feature_columns = joblib.load('feature_columns.pkl')

# Jo 11 fields form mein nahi hain, unki fixed default values
DEFAULTS = {
    'CholCheck': 1, 'Stroke': 0, 'Fruits': 1, 'Veggies': 1,
    'HvyAlcoholConsump': 0, 'AnyHealthcare': 1, 'NoDocbcCost': 0,
    'MentHlth': 0, 'PhysHlth': 0, 'Education': 5, 'Income': 6,
}


def age_to_group(age):
    """Asli age ko dataset ke age group (1 se 13) mein badalta hai."""
    if age <= 24:
        return 1
    if age >= 80:
        return 13
    return (age - 25) // 5 + 2


def risk_level(prob):
    """Model ke score ko Low / Moderate / High mein badalta hai."""
    if prob < 0.30:
        return 'Low'
    if prob < 0.60:
        return 'Moderate'
    return 'High'


@app.route('/', methods=['GET', 'POST'])
def home():
    result = None
    error = None

    if request.method == 'POST':
        try:
            age = int(request.form['age'])
            bmi = float(request.form['bmi'])
            if not (18 <= age <= 100) or not (10 <= bmi <= 80):
                raise ValueError

            row = {
                'Age': age_to_group(age),
                'Sex': int(request.form['sex']),
                'BMI': bmi,
                'HighBP': int(request.form['high_bp']),
                'HighChol': int(request.form['high_chol']),
                'GenHlth': int(request.form['gen_hlth']),
                'PhysActivity': int(request.form['phys_activity']),
                'Smoker': int(request.form['smoker']),
                'HeartDiseaseorAttack': int(request.form['heart_disease']),
                'DiffWalk': int(request.form['diff_walk']),
            }
            row.update(DEFAULTS)

            # Columns ko bilkul training wale order mein lagao
            X = pd.DataFrame([row])[feature_columns]
            prob = model.predict_proba(X)[0][1]

            result = {
                'level': risk_level(prob),
                'score': round(prob * 100),
            }
        except (ValueError, KeyError):
            error = 'Please fill all fields with valid values (age 18-100, BMI 10-80).'

    return render_template('index.html', result=result, error=error)


if __name__ == '__main__':
    app.run(debug=True)
from flask import Flask, request, jsonify
import joblib
import numpy as np
import pandas as pd

# Load saved pipeline
model = joblib.load('rf_model_pipeline.pkl')

log_cols = ['person_age', 'person_income', 'loan_amnt', 'loan_percent_income']

app = Flask(__name__)

@app.route('/')
def home():
    return "Loan Prediction API is running!"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        input_df = pd.DataFrame([data])

        # Apply log1p to the relevant columns
        for col in log_cols:
            input_df[col] = np.log1p(input_df[col])

        # Make prediction
        prediction = model.predict(input_df)[0]

        return jsonify({
            'prediction': int(prediction),
            'risk_level': 'High Risk' if prediction == 0 else 'Low Risk'
        })

    except Exception as e:
        return jsonify({'error': str(e)})

from flask import Flask, request, render_template_string
import pandas as pd

app = Flask(__name__)

@app.route('/form', methods=['GET', 'POST'])
def form():
    prediction = None
    risk = None
    color = None

    if request.method == 'POST':
        data = {
            'person_age': float(request.form['person_age']),
            'person_education': request.form['person_education'],
            'person_income': float(request.form['person_income']),
            'person_home_ownership': request.form['person_home_ownership'],
            'loan_amnt': float(request.form['loan_amnt']),
            'loan_intent': request.form['loan_intent'],
            'loan_int_rate': float(request.form['loan_int_rate']),
            'loan_percent_income': float(request.form['loan_percent_income']),
            'credit_score': float(request.form['credit_score']),
            'previous_loan_defaults_on_file': request.form['previous_loan_defaults_on_file']
        }

        df = pd.DataFrame([data])

        # 🔶 Apply log1p on required numeric columns
        for col in ['person_age', 'person_income', 'loan_amnt', 'loan_percent_income']:
            df[col] = np.log1p(df[col])

        # 🔷 Predict
        pred = model.predict(df)[0]

        risk = "Low Risk" if pred == 1 else "High Risk"
        prediction = risk
        color = "green" if pred == 1 else "red"

    return render_template_string('''
<html>
<head>
    <title>Credit Risk Form</title>
    <style>
        body {
            margin: 0;
            padding: 0;
            background-color: #1f2a40; /* Navy blue background */
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            color: #ffffff;
        }
        .container {
            max-width: 700px;
            margin: 50px auto;
            background-color: #27344f; /* Darker navy for card */
            padding: 40px;
            border-radius: 15px;
            box-shadow: 0 0 15px rgba(0, 0, 0, 0.25);
        }
        h1 {
            text-align: center;
            color: #ff9800; /* Orange */
            margin-bottom: 30px;
        }
        label {
            display: block;
            margin-bottom: 5px;
            color: #ffcc80; /* Light orange */
        }
        input, select {
            width: 100%;
            padding: 10px;
            border: none;
            border-radius: 5px;
            margin-bottom: 20px;
            font-size: 14px;
        }
        input[type="submit"] {
            background-color: #ff9800;
            color: #fff;
            font-weight: bold;
            cursor: pointer;
            transition: background-color 0.3s ease;
        }
        input[type="submit"]:hover {
            background-color: #e67e00;
        }
        .result {
            text-align: center;
            margin-top: 30px;
            padding-top: 20px;
            border-top: 1px solid #ff9800;
        }
        .result h2 {
            font-size: 24px;
        }
        .result p {
            font-size: 18px;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>Automated Credit Risk Modelling</h1>
        <form method="POST">
            <label>Age:</label>
            <input type="number" name="person_age" required>

            <label>Education:</label>
            <select name="person_education" required>
                <option>Bachelor</option><option>Master</option><option>High School</option>
                <option>Associate</option><option>Doctorate</option>
            </select>

            <label>Income:</label>
            <input type="number" name="person_income" required>

            <label>Ownership:</label>
            <select name="person_home_ownership" required>
                <option>RENT</option><option>OWN</option><option>MORTGAGE</option><option>OTHER</option>
            </select>

            <label>Loan Amount:</label>
            <input type="number" name="loan_amnt" required>

            <label>Loan Intent:</label>
            <select name="loan_intent" required>
                <option>EDUCATION</option><option>VENTURE</option><option>MEDICAL</option>
                <option>DEBTCONSOLIDATION</option><option>HOMEIMPROVEMENT</option><option>PERSONAL</option>
            </select>

            <label>Interest Rate:</label>
            <input type="number" step="0.01" name="loan_int_rate" required>

            <label>Percent Income:</label>
            <input type="number" step="0.01" name="loan_percent_income" required>

            <label>Credit Score:</label>
            <input type="number" name="credit_score" required>

            <label>Defaults:</label>
            <select name="previous_loan_defaults_on_file" required>
                <option>No</option><option>Yes</option>
            </select>

            <input type="submit" value="Submit">
        </form>

        {% if prediction %}
        <div class="result">
            <h2 style="color: {{ color }};"><strong>Prediction:</strong> {{ prediction }}</h2>
        </div>
        {% endif %}
    </div>
</body>
</html>
    ''', prediction=prediction, color=color)



if __name__ == '__main__':
    app.run(debug=True)




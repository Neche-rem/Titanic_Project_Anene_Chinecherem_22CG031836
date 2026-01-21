from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load trained model
model = joblib.load('model/titanic_survival_model.pkl')

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = ''
    if request.method == 'POST':
        try:
            # Get input values from form
            pclass = int(request.form['pclass'])
            sex = request.form['sex']
            age = float(request.form['age'])
            sibsp = int(request.form['sibsp'])
            fare = float(request.form['fare'])

            # Encode Sex: male=1, female=0
            sex_encoded = 1 if sex == 'male' else 0

            # Scale Age & Fare using the same formula you used when training
            # Approximate mean & std from Titanic dataset
            age_scaled = (age - 29.7) / 14.5
            fare_scaled = (fare - 32.2) / 49.7

            # Create feature array
            input_data = np.array([[pclass, sex_encoded, age_scaled, sibsp, fare_scaled]])

            # Make prediction
            pred = model.predict(input_data)[0]
            prediction = 'Survived' if pred == 1 else 'Did Not Survive'

        except Exception as e:
            prediction = f"Error: {e}"

    return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    import os
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port, debug=True)

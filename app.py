from flask import Flask, render_template, request
import pickle
import numpy as np

# Load trained model
with open("disease_model.pkl", "rb") as f:
    model = pickle.load(f)

app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get user input
        age = int(request.form["age"])
        blood_pressure = int(request.form["blood_pressure"])
        cholesterol = int(request.form["cholesterol"])
        glucose = int(request.form["glucose"])
        body_temperature = float(request.form["body_temperature"])  # New field added

        # Create input array for model
        features = np.array([[age, blood_pressure, cholesterol, glucose, body_temperature]])

        # Predict using the trained model
        prediction = model.predict(features)[0]

        # Determine disease status
        result = "Positive (Disease Detected)" if prediction == 1 else "Negative (No Disease)"

        return render_template("result.html", prediction=result)

    except Exception as e:
        return f"Error: {e}"

if __name__ == "__main__":
    app.run(debug=True)
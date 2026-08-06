from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load model and scaler
model = joblib.load("models/random_forest_model.pkl")
scaler = joblib.load("models/scaler.pkl")

print("Model Loaded Successfully")
print("Expected Features:", model.n_features_in_)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    try:

        features = [

            float(request.form["completion_rate"]),
            float(request.form["dropout_rate"]),
            float(request.form["avg_score_improvement"]),
            float(request.form["avg_quiz_score"]),
            float(request.form["avg_watch_time"]),
            float(request.form["assignment_submission_rate"]),
            float(request.form["forum_activity_rate"]),
            float(request.form["avg_feedback_score"]),
            float(request.form["feedback_response_rate"])

        ]

        # If model expects extra feature(s), add them here.
        while len(features) < model.n_features_in_:
            features.append(1)      # Default value (change if required)

        features = np.array([features])

        print("Input Shape :", features.shape)
        print("Input Data :", features)

        features_scaled = scaler.transform(features)

        prediction = model.predict(features_scaled)

        print("Prediction :", prediction)

        return render_template(
            "index.html",
            prediction=prediction[0]
        )

    except Exception as e:

        print("ERROR :", e)

        return render_template(
            "index.html",
            prediction=f"Error : {e}"
        )


if __name__ == "__main__":
    app.run(debug=True)
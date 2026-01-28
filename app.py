from flask import Flask, request, jsonify
import joblib
import os

app = Flask(__name__)
model = joblib.load("model.pkl")

@app.route("/")
def home():
    return "ML Model is Running!"

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json["value"]
    prediction = model.predict([[data]])[0]
    return jsonify({"prediction": int(prediction)})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))   # Render uses PORT
    app.run(host="0.0.0.0", port=port)

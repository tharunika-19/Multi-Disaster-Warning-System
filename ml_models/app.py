
from flask import Flask, request, jsonify
import pickle
import numpy as np
import gdown
import os

app = Flask(__name__)

# Download models from Google Drive
def download_models():
    if not os.path.exists("earthquake_model.pkl"):
        gdown.download("https://drive.google.com/uc?id=1zGgZq7mO5QmqYg8QQvv7qd_4wfwZRZ-N", "earthquake_model.pkl", quiet=False)
    if not os.path.exists("cyclone_model.pkl"):
        gdown.download("https://drive.google.com/uc?id=1Ey170-XHSdf-r1jfRUYf9fAv8d2-R2Mk", "cyclone_model.pkl", quiet=False)
    if not os.path.exists("flood_model.pkl"):
        gdown.download("https://drive.google.com/uc?id=1wcscT46u4pWVOi2TSF9BYqfoara-Oi_E", "flood_model.pkl", quiet=False)

download_models()

# Load models
with open("earthquake_model.pkl", "rb") as f:
    earthquake_model = pickle.load(f)
with open("flood_model.pkl", "rb") as f:
    flood_model = pickle.load(f)
with open("cyclone_model.pkl", "rb") as f:
    cyclone_model = pickle.load(f)

print("All models loaded!")

@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "Disaster Prediction API is running!"})

@app.route("/predict/earthquake", methods=["POST"])
def predict_earthquake():
    data = request.json
    features = np.array([[data["magnitude"], data["depth"],
                          data["tsunami"], data["significance"],
                          data["latitude"], data["longitude"]]])
    result = earthquake_model.predict(features)[0]
    return jsonify({"risk": int(result), "message": "High Risk" if result == 1 else "Low Risk"})

@app.route("/predict/flood", methods=["POST"])
def predict_flood():
    data = request.json
    features = np.array([[data["MonsoonIntensity"], data["TopographyDrainage"],
                          data["RiverManagement"], data["Deforestation"],
                          data["Urbanization"], data["ClimateChange"],
                          data["DamsQuality"], data["Siltation"],
                          data["AgriculturalPractices"], data["Encroachments"],
                          data["IneffectiveDisasterPreparedness"], data["DrainageSystems"],
                          data["CoastalVulnerability"], data["Landslides"],
                          data["Watersheds"], data["DeterioratingInfrastructure"],
                          data["PopulationScore"], data["WetlandLoss"],
                          data["InadequatePlanning"], data["PoliticalFactors"]]])
    result = flood_model.predict(features)[0]
    return jsonify({"risk": int(result), "message": "High Risk" if result == 1 else "Low Risk"})

@app.route("/predict/cyclone", methods=["POST"])
def predict_cyclone():
    data = request.json
    features = np.array([[data["pressure"], data["latitude"], data["longitude"]]])
    result = cyclone_model.predict(features)[0]
    return jsonify({"risk": int(result), "message": "High Risk" if result == 1 else "Low Risk"})

if __name__ == "__main__":
    app.run(debug=True)

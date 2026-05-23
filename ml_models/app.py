
from flask import Flask, request, jsonify
import pickle
import numpy as np

app = Flask(__name__)

# Load models
with open("earthquake_model.pkl", "rb") as f:
    earthquake_model = pickle.load(f)

with open("flood_model.pkl", "rb") as f:
    flood_model = pickle.load(f)

with open("cyclone_model.pkl", "rb") as f:
    cyclone_model = pickle.load(f)

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

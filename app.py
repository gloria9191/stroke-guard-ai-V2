from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# ---- 모델 로드 ----
with open("stroke_model_v2.pkl", "rb") as f:
    package = pickle.load(f)

model = package["model"]
scaler = package["scaler"]
feature_cols = package["feature_cols"]
cluster_centers = package["cluster_centers"]

# ---- 위험도 구간 labeling ----
def get_risk_label(score):
    if score >= 0.20:
        return "High Risk"
    elif score >= 0.07:
        return "Moderate Risk"
    else:
        return "Low Risk"

# ---- cluster 매칭 ----
def closest_cluster(x_scaled):
    dists = np.linalg.norm(cluster_centers - x_scaled, axis=1)
    return np.argmin(dists)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    # 입력값
    data = []
    for col in feature_cols:
        val = float(request.form.get(col, 0))
        data.append(val)

    # numpy
    X = np.array(data).reshape(1, -1)
    X_scaled = scaler.transform(X)

    # 회귀 위험도 예측
    risk_score = float(model.predict(X_scaled)[0])

    # 리스크 등급
    risk_label = get_risk_label(risk_score)

    # cluster 예측
    cluster_id = closest_cluster(X_scaled)

    return render_template(
        "result.html",
        score=round(risk_score, 4),
        label=risk_label,
        cluster=cluster_id
    )

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)

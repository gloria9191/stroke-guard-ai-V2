import pickle
import numpy as np
from flask import Flask, request, jsonify

app = Flask(__name__)

# ---------------------------
# 1) 모델 로딩
# ---------------------------
with open("stroke_model_v2.pkl", "rb") as f:
    pkg = pickle.load(f)

model = pkg["model"]
scaler = pkg["scaler"]
feature_cols = pkg["feature_cols"]
cluster_centers = pkg["cluster_centers"]


# ---------------------------
# 2) 리스크 등급 함수
# ---------------------------
def risk_level(x):
    if x < 0.02:
        return "매우 낮음"
    elif x < 0.05:
        return "낮음"
    elif x < 0.10:
        return "중간"
    elif x < 0.20:
        return "높음"
    else:
        return "매우 높음"


# ---------------------------
# 3) 클러스터 계산
# ---------------------------
def find_cluster(x_scaled):
    dists = np.linalg.norm(cluster_centers - x_scaled, axis=1)
    return int(np.argmin(dists))


# ---------------------------
# 4) 메인페이지 HTML 직접 렌더
# ---------------------------
@app.route("/")
def index():
    return """
<!DOCTYPE html>
<html lang="ko">
<head>
    <meta charset="UTF-8">
    <title>StrokeGuard AI V2</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/css/bootstrap.min.css" rel="stylesheet">
</head>

<body class="bg-light">

<div class="container py-5">
    <h1 class="mb-4 text-center fw-bold">🧠 StrokeGuard AI – 뇌졸중 위험도 예측</h1>

    <div class="card shadow p-4">

        <form id="predictForm">

            <div class="row mb-3">
                <div class="col">
                    <label class="form-label">나이(Age)</label>
                    <input type="number" class="form-control" name="Age" required>
                </div>
                <div class="col">
                    <label class="form-label">성별(Sex) — 1:남, 2:여</label>
                    <input type="number" class="form-control" name="Sex" required>
                </div>
            </div>

            <div class="row mb-3">
                <div class="col">
                    <label class="form-label">BMI</label>
                    <input type="number" class="form-control" name="BMI" required>
                </div>
                <div class="col">
                    <label class="form-label">SBP(수축기혈압)</label>
                    <input type="number" class="form-control" name="SBP_mean" required>
                </div>
                <div class="col">
                    <label class="form-label">DBP(이완기혈압)</label>
                    <input type="number" class="form-control" name="DBP_mean" required>
                </div>
            </div>

            <div class="mb-3">
                <label class="form-label">공복혈당(Glucose)</label>
                <input type="number" class="form-control" name="Glucose" required>
            </div>

            <button type="submit" class="btn btn-primary w-100">
                위험도 예측하기
            </button>
        </form>

        <hr class="my-4">

        <h4>📊 결과</h4>
        <div id="resultBox"></div>

    </div>
</div>

<script>
document.getElementById("predictForm").addEventListener("submit", async function(e){
    e.preventDefault();

    const formData = new FormData(this);
    const response = await fetch("/predict", {
        method: "POST",
        body: formData
    });

    const data = await response.json();

    if(data.error){
        document.getElementById("resultBox").innerHTML =
            `<div class="alert alert-danger">${data.error}</div>`;
        return;
    }

    document.getElementById("resultBox").innerHTML = `
        <div class="alert alert-info">
            <h5>예측된 뇌졸중 위험도: <strong>${(data.risk * 100).toFixed(2)}%</strong></h5>
            <p>위험 등급: <strong>${data.risk_label}</strong></p>
            <p>군집 Cluster ID: <strong>${data.cluster}</strong></p>
        </div>
    `;
});
</script>

</body>
</html>
"""


# ---------------------------
# 5) 예측 API
# ---------------------------
@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.form if request.form else request.json

        age = float(data.get("Age"))
        sex = float(data.get("Sex"))
        bmi = float(data.get("BMI"))
        sbp = float(data.get("SBP_mean"))
        dbp = float(data.get("DBP_mean"))
        glucose = float(data.get("Glucose"))

        # 모델 입력 정렬
        x = np.array([[age, sex, bmi, sbp, dbp, glucose]])

        # 스케일링
        x_scaled = scaler.transform(x)

        # 예측
        risk = float(model.predict(x_scaled)[0])

        # 클러스터
        cluster = find_cluster(x_scaled)

        return jsonify({
            "risk": risk,
            "risk_label": risk_level(risk),
            "cluster": cluster
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 400


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)

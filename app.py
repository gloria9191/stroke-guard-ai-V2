from flask import Flask, request, render_template_string, jsonify
import joblib
import numpy as np

app = Flask(__name__)
model = joblib.load("stroke_model.pkl")
scaler = joblib.load("scaler.pkl")
feature_names = joblib.load("feature_names.pkl")

HTML = """
<!DOCTYPE html>
<html lang="ko">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>StrokeGuard AI - 뇌졸중 조기 경보 시스템</title>

    <!-- Fonts & Icons -->
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/css/bootstrap.min.css" rel="stylesheet">
    <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.5.0/css/all.min.css" rel="stylesheet">
    <link href="https://fonts.googleapis.com/css2?family=Nanum+Myeongjo:wght@700&family=Noto+Sans+KR:wght@300;400;700&display=swap" rel="stylesheet">

    <style>
        body {
            font-family: 'Noto Sans KR', sans-serif;
            background: linear-gradient(135deg,#667eea,#764ba2);
            min-height: 100vh;
        }

        /* Hero Section */
        .hero {
            background: linear-gradient(rgba(0,0,0,0.75),rgba(0,0,0,0.75)),
                        url('https://images.unsplash.com/photo-1576091160399-112ba8d25d1d?auto=format&fit=crop&q=80') center/cover;
            color: white;
            padding: 140px 20px;
            text-align: center;
            border-radius: 0 0 45px 45px;
            box-shadow: 0px 10px 40px rgba(0,0,0,0.3);
        }

        .hero h1 {
            font-size: 4rem;
            font-weight: 800;
            letter-spacing: -1px;
        }

        /* Card Upgrade */
        .card-custom {
            border-radius: 30px;
            backdrop-filter: blur(14px);
            -webkit-backdrop-filter: blur(14px);
            background: rgba(255,255,255,0.88);
            border: 1px solid rgba(255,255,255,0.35);
            box-shadow: 0 25px 60px rgba(0,0,0,0.25);
            transition: 0.4s;
        }

        .card-custom:hover {
            transform: translateY(-12px);
            box-shadow: 0 35px 70px rgba(0,0,0,0.35);
        }

        /* Predict Button */
        .btn-predict {
            background: linear-gradient(45deg,#e74c3c,#c0392b);
            border: none;
            border-radius: 50px;
            padding: 18px 70px;
            font-size: 1.5rem;
            font-weight: bold;
            color: white;
            letter-spacing: 1px;
            cursor: pointer;
            box-shadow: 0 12px 30px rgba(231,76,60,0.5);
            transition: 0.25s;
        }

        .btn-predict:hover {
            box-shadow: 0 18px 40px rgba(231,76,60,0.7);
            transform: translateY(-4px);
        }

        /* Pulse Animation */
        .pulse {animation:pulse 2s infinite}
        @keyframes pulse{
            0%{box-shadow:0 0 0 0 rgba(231,76,60,0.7)}
            70%{box-shadow:0 0 0 25px rgba(231,76,60,0)}
            100%{box-shadow:0 0 0 0 rgba(231,76,60,0)}
        }

        /* Result Card */
        .result-high{
            background:linear-gradient(135deg,#ff6b6b,#ee5a52);
            color:white;
        }
        .result-low{
            background:linear-gradient(135deg,#51cf66,#40c057);
            color:white;
        }

        /* Fade-in Animation */
        .fadeSmooth {
            animation: fadeSmooth 1s ease forwards;
        }
        @keyframes fadeSmooth {
            from { opacity: 0; transform: translateY(20px); }
            to   { opacity: 1; transform: translateY(0); }
        }

    </style>
</head>

<body>

<!-- HERO -->
<div class="hero">
    <h1>StrokeGuard AI</h1>
    <p class="lead fs-3 mt-3">인공지능 기반 뇌졸중 조기 예측 시스템</p>
    <p class="fs-4 opacity-90 mt-2">
        제1회 전국 데이터 분석 대회 <span class="text-warning fw-bold">대상 수상작</span>
    </p>
</div>

<!-- FORM -->
<div class="container my-5">
    <div class="row justify-content-center">
        <div class="col-lg-10">
            <div class="card-custom p-5">
                <h2 class="text-center mb-5 text-danger fw-bold">
                    <i class="fas fa-heartbeat me-2"></i>환자 정보 입력
                </h2>

                <form id="form" class="row g-4">
                    <div class="col-md-4"><label>나이</label><input type="number" class="form-control form-control-lg" id="Age" value="78"></div>
                    <div class="col-md-4"><label>BMI</label><input type="number" step="0.1" class="form-control form-control-lg" id="BMI" value="32.5"></div>
                    <div class="col-md-4"><label>혈당</label><input type="number" step="0.1" class="form-control form-control-lg" id="Glucose" value="185"></div>

                    <div class="col-md-4"><label>수축기 혈압</label><input type="number" class="form-control form-control-lg" id="SBP_mean" value="168"></div>
                    <div class="col-md-4"><label>이완기 혈압</label><input type="number" class="form-control form-control-lg" id="DBP_mean" value="92"></div>

                    <div class="col-md-2"><label>고혈압</label>
                        <select class="form-select form-select-lg" id="Hypertension">
                            <option value="1" selected>있음</option>
                            <option value="0">없음</option>
                        </select>
                    </div>

                    <div class="col-md-2"><label>당뇨</label>
                        <select class="form-select form-select-lg" id="Diabetes">
                            <option value="1" selected>있음</option>
                            <option value="0">없음</option>
                        </select>
                    </div>

                    <div class="col-12 text-center mt-5">
                        <button type="submit" class="btn-predict pulse">AI 예측 시작</button>
                    </div>
                </form>
            </div>
        </div>
    </div>

    <!-- RESULT -->
    <div class="row justify-content-center mt-5" id="result" style="display:none;">
        <div class="col-lg-8">
            <div class="card-custom p-5 text-center rounded-4" id="resultCard">
                <h1 class="display-1 fw-bold mb-4" id="prob">0%</h1>
                <h2 id="level" class="mb-4 fw-bold"></h2>
                <canvas id="gaugeChart" width="250" height="130"></canvas>
                <p class="fs-4 mt-4" id="advice"></p>
            </div>
        </div>
    </div>
</div>

<!-- SCRIPTS -->
<script src="https://code.jquery.com/jquery-3.6.0.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/chart.js"></script>

<script>
let chart;

$("#form").submit(function(e) {
    e.preventDefault();

    const d = {
        Age: +$("#Age").val(), 
        BMI: +$("#BMI").val(), 
        Glucose: +$("#Glucose").val(),
        SBP_mean: +$("#SBP_mean").val(), 
        DBP_mean: +$("#DBP_mean").val(),
        Hypertension: +$("#Hypertension").val(), 
        Diabetes: +$("#Diabetes").val()
    };

    fetch("/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(d)
    })
    .then(r => r.json())
    .then(res => {
        const prob = (res.prob * 100).toFixed(1);
        $("#prob").text(prob + "%");

        if (res.prob > 0.5) {
            $("#resultCard").removeClass("result-low").addClass("result-high");
            $("#level").html("고위험군 <i class='fas fa-exclamation-triangle'></i>");
            $("#advice").text("즉시 병원 방문을 강력 권고합니다!");
        } else {
            $("#resultCard").removeClass("result-high").addClass("result-low");
            $("#level").html("안전군 <i class='fas fa-shield-alt'></i>");
            $("#advice").text("정기 검진을 유지하세요!");
        }

        $("#result").fadeIn(700).addClass("fadeSmooth");

        if (chart) chart.destroy();

        chart = new Chart($("#gaugeChart"), {
            type: 'doughnut',
            data: {
                datasets: [{
                    data: [prob, 100-prob],
                    backgroundColor: [
                        res.prob > 0.5 ? '#e74c3c' : '#51cf66',
                        '#e9ecef'
                    ],
                    cutout: '80%'
                }]
            },
            options: { 
                plugins: { legend: { display: false } }
            }
        });
    })
    .catch(() => {
        alert("⚠ 예측 중 오류가 발생했습니다. 입력 값을 다시 확인해주세요.");
    });
});
</script>

</body>
</html>
"""

@app.route("/")
def home():
    return render_template_string(HTML)

@app.route("/predict", methods=["POST"])
def predict():
    d = request.get_json()
    row = np.zeros(len(feature_names))

    for i, col in enumerate(feature_names):
        if col == "SEQN":
            row[i] = 92000
        elif col == "bmi_glu":
            row[i] = float(d["BMI"]) * float(d["Glucose"])
        elif col == "age_sbp":
            row[i] = float(d["Age"]) * float(d["SBP_mean"])
        elif col == "htn_dm":
            row[i] = float(d["Hypertension"]) + float(d["Diabetes"])
        elif col in d:
            row[i] = float(d[col])

    prob = model.predict_proba(scaler.transform([row]))[0][1]
    prob = prob ** 0.25

    return jsonify({"prob": float(prob)})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)



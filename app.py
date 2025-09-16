from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Flutter 연결 시 CORS 허용

@app.route("/")
def home():
    return jsonify({"message": "서버 연결 성공!"})

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    text = data.get("text", "")
    # 여기에 실제 모델 로직 넣으면 됨
    return jsonify({"result": f"'{text}'를 받았습니다."})

if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)


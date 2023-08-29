from flask import Flask, request, jsonify
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from flask_cors import CORS  # flask_cors를 import

app = Flask(__name__)
CORS(app)  # CORS를 Flask 앱에 적용

# 모델 초기화
knn = KNeighborsClassifier()

# 모델 학습 및 초기화 함수
def initialize_model():
    global scaler
    dataset = pd.read_csv('../심혈관질환.csv')
    X = dataset[['SBP', 'DBP', 'BMI']]
    y = dataset['HCVD']
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    knn.fit(X, y)

# 초기 모델 학습
initialize_model()

@app.route('/predict', methods=['POST'])
def predict():
    sbp = float(request.json['sbp'])
    dbp = float(request.json['dbp'])
    height = float(request.json['height'])
    weight = float(request.json['weight'])
    bmi = weight / ((height * 0.01) ** 2)
    input_data = scaler.transform([[sbp, dbp, bmi]])
    prediction = knn.predict(input_data)[0]
    if prediction == 1:
        result_message = '심혈관 질환 검사를 권장합니다.'
    else:
        result_message = '심혈관 질환 위험군에 속하지는 않지만 꾸준한 관리는 필수입니다!'
    return jsonify({'prediction': result_message})

if __name__ == '__main__':
    app.run(debug=True)

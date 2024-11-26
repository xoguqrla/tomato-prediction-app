import pandas as pd
import numpy as np
import streamlit as st
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# 데이터 로드 및 전처리
def load_and_prepare_data(file_path):
    try:
        # 데이터 로드
        data = pd.read_csv(file_path, encoding='utf-8-sig')
        columns = ['개화군', '착과군', '초장', '생장길이', '엽수', '엽장', '엽폭', '줄기굵기', '화방높이', '화방별꽃수']
        data = data[columns].dropna()  # 필요한 열만 선택하고 결측값 제거
        return data
    except FileNotFoundError:
        st.error(f"파일을 찾을 수 없습니다: {file_path}")
        return None

# 모델 학습
def train_models(data):
    # 개화군 예측 모델
    X_bloom = data[['초장', '생장길이', '엽수', '엽장', '엽폭', '줄기굵기', '화방높이', '화방별꽃수']]
    y_bloom = data['개화군']
    bloom_model = LinearRegression()
    bloom_model.fit(X_bloom, y_bloom)

    # 착과군 예측 모델
    X_fruit = data[['개화군']]
    y_fruit = data['착과군']
    fruit_model = LinearRegression()
    fruit_model.fit(X_fruit, y_fruit)

    return bloom_model, fruit_model

# 사용자 입력 기반 예측
def predict_with_user_input(bloom_model, fruit_model, data):
    st.header("생육 데이터 입력")
    mean_values = data[['초장', '생장길이', '엽수', '엽장', '엽폭', '줄기굵기', '화방높이', '화방별꽃수']].mean()

    user_input = {}
    for feature, example in mean_values.items():
        user_input[feature] = st.number_input(f"{feature} (예: {round(example, 2)})", value=round(example, 2))

    if st.button("예측 실행"):
        # 입력 데이터를 데이터프레임으로 변환
        input_df = pd.DataFrame([user_input])

        # 개화군 예측
        bloom_count = bloom_model.predict(input_df)[0]

        # 착과군 예측
        fruit_count = fruit_model.predict([[bloom_count]])[0]

        # 결과 출력
        st.header("예측 결과")
        st.write(f"예측된 개화군 수: **{bloom_count:.2f}**")
        st.write(f"예측된 착과군 수: **{fruit_count:.2f}**")

        # 평가 메트릭 계산
        calculate_metrics(bloom_model, fruit_model, data)

# 평가 메트릭 계산 (오차율 및 신뢰도)
def calculate_metrics(bloom_model, fruit_model, data):
    st.header("모델 평가")
    # 개화군 예측 신뢰도
    X_bloom = data[['초장', '생장길이', '엽수', '엽장', '엽폭', '줄기굵기', '화방높이', '화방별꽃수']]
    y_bloom = data['개화군']
    bloom_pred = bloom_model.predict(X_bloom)

    bloom_r2 = r2_score(y_bloom, bloom_pred)
    bloom_mape = np.mean(np.abs((y_bloom - bloom_pred) / y_bloom)) * 100

    # 착과군 예측 신뢰도
    X_fruit = data[['개화군']]
    y_fruit = data['착과군']
    fruit_pred = fruit_model.predict(X_fruit)

    fruit_r2 = r2_score(y_fruit, fruit_pred)
    fruit_mape = np.mean(np.abs((y_fruit - fruit_pred) / y_fruit)) * 100

    # 결과 출력
    st.write(f"[개화군 예측] R² (신뢰도): **{bloom_r2:.2f}**, MAPE (평균 절대 오차율): **{bloom_mape:.2f}%**")
    st.write(f"[착과군 예측] R² (신뢰도): **{fruit_r2:.2f}**, MAPE (평균 절대 오차율): **{fruit_mape:.2f}%**")

# Streamlit 애플리케이션 실행
def main():
    st.title("토마토 생육 예측 시스템")
    st.write("생육 데이터를 입력하여 개화군과 착과군을 예측하고, 모델 성능을 평가합니다.")

    # 파일 경로 설정
    file_path = r"C:\Users\k7202\OneDrive\바탕 화면\완숙토마토_생육_데이터.csv"

    # 데이터 로드
    data = load_and_prepare_data(file_path)
    if data is None:
        return

    st.write("데이터 로드 완료. 데이터 미리 보기:")
    st.write(data.head())

    # 모델 학습
    bloom_model, fruit_model = train_models(data)
    st.write("모델 학습 완료.")

    # 사용자 입력 기반 예측
    predict_with_user_input(bloom_model, fruit_model, data)

if __name__ == "__main__":
    main()

import pandas as pd
import numpy as np
import streamlit as st
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# 데이터 로드 및 전처리
def load_and_prepare_data(file_path):
    try:
        data = pd.read_csv(file_path, encoding='utf-8-sig')
        columns = ['개화군', '착과군', '초장', '생장길이', '엽수', '엽장', '엽폭', '줄기굵기', '화방높이', '화방별꽃수']
        data = data[columns].dropna()
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
    st.sidebar.title("목차")
    options = st.sidebar.radio("메뉴를 선택하세요:", ["토마토 소개", "토마토 생육 예측 시스템", "환경 데이터 분석"])

    if options == "토마토 소개":
        st.title("토마토 소개")
        st.image("https://upload.wikimedia.org/wikipedia/commons/8/89/Tomato_je.jpg", caption="토마토 이미지")
        st.write("""
            토마토는 전 세계적으로 인기 있는 작물로, 다양한 기후와 환경에서 재배됩니다.
            이 앱은 토마토 생육 데이터를 기반으로 개화군과 착과군의 예측 및 환경 분석을 제공합니다.
        """)

    elif options == "토마토 생육 예측 시스템":
        st.title("토마토 생육 예측 시스템")
        st.write("생육 데이터를 입력하여 개화군과 착과군을 예측하고, 모델 성능을 평가합니다.")

        file_path = "완숙토마토_생육_데이터.csv"
        data = load_and_prepare_data(file_path)
        if data is None:
            return

        bloom_model, fruit_model = train_models(data)
        predict_with_user_input(bloom_model, fruit_model, data)

    elif options == "환경 데이터 분석":
        st.title("환경 데이터 분석")
        st.write("계절별 최적 환경 탐색 및 월별 환경 평가 결과를 제공합니다.")

        file_path = "토마토_환경데이터_익산_1구역(1월~11월).csv"
        data, numeric_data = load_and_process_data(file_path)
        if data is None or numeric_data is None:
            return

        seasonal_stats = analyze_seasonal_conditions(numeric_data, data)
        st.write("### 계절별 최적 환경 데이터")
        st.write(seasonal_stats)

        optimal_conditions = seasonal_stats.to_dict('index')
        monthly_deviation = evaluate_monthly_conditions(numeric_data, data, optimal_conditions)
        st.write("### 월별 환경 데이터 평가")
        st.write(monthly_deviation)

if __name__ == "__main__":
    main()

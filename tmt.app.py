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

# 환경 데이터 로드 및 전처리
def load_and_process_data(file_path):
    try:
        data = pd.read_csv(file_path, encoding='utf-8-sig')
        data['저장시각'] = pd.to_datetime(data['저장시각'], format='%Y.%m.%d %H:%M')
        data['월'] = data['저장시각'].dt.month
        data['계절'] = data['월'].apply(lambda x: '봄' if x in [3, 4, 5] else
                                        '여름' if x in [6, 7, 8] else
                                        '가을' if x in [9, 10, 11] else
                                        '겨울')
        return data, data.select_dtypes(include=['float64', 'int64'])
    except Exception as e:
        st.error(f"데이터 처리 중 오류 발생: {e}")
        return None, None

# 모델 학습
def train_models(data):
    X_bloom = data[['초장', '생장길이', '엽수', '엽장', '엽폭', '줄기굵기', '화방높이', '화방별꽃수']]
    y_bloom = data['개화군']
    bloom_model = LinearRegression()
    bloom_model.fit(X_bloom, y_bloom)

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
        input_df = pd.DataFrame([user_input])
        bloom_count = bloom_model.predict(input_df)[0]
        fruit_count = fruit_model.predict([[bloom_count]])[0]
        st.write(f"예측된 개화군 수: **{bloom_count:.2f}**")
        st.write(f"예측된 착과군 수: **{fruit_count:.2f}**")

        # 모델 평가
        calculate_metrics(bloom_model, fruit_model, data)

# 평가 메트릭 계산
def calculate_metrics(bloom_model, fruit_model, data):
    X_bloom = data[['초장', '생장길이', '엽수', '엽장', '엽폭', '줄기굵기', '화방높이', '화방별꽃수']]
    y_bloom = data['개화군']
    bloom_pred = bloom_model.predict(X_bloom)
    bloom_r2 = r2_score(y_bloom, bloom_pred)
    bloom_mape = np.mean(np.abs((y_bloom - bloom_pred) / y_bloom)) * 100

    X_fruit = data[['개화군']]
    y_fruit = data['착과군']
    fruit_pred = fruit_model.predict(X_fruit)
    fruit_r2 = r2_score(y_fruit, fruit_pred)
    fruit_mape = np.mean(np.abs((y_fruit - fruit_pred) / y_fruit)) * 100

    st.write(f"[개화군 예측] R²: **{bloom_r2:.2f}**, MAPE: **{bloom_mape:.2f}%**")
    st.write(f"[착과군 예측] R²: **{fruit_r2:.2f}**, MAPE: **{fruit_mape:.2f}%**")

# 메인 페이지
def main():
    st.sidebar.title("목차")
    page = st.sidebar.radio("페이지를 선택하세요", ["토마토 소개", "생육 예측 시스템", "환경 데이터 분석"])

    if page == "토마토 소개":
        st.title("토마토 소개")
        st.image("https://upload.wikimedia.org/wikipedia/commons/8/89/Tomato_je.jpg", caption="토마토")
        st.write("""
            토마토는 전 세계적으로 사랑받는 작물로, 영양가가 높고 다양한 요리에 사용됩니다.
            이 앱은 토마토의 생육 데이터를 활용해 개화군과 착과군을 예측하고, 최적의 환경을 분석하는 도구입니다.
        """)

    elif page == "생육 예측 시스템":
        st.title("토마토 생육 예측 시스템")
        st.write("생육 데이터를 입력하여 개화군과 착과군을 예측하고, 모델 성능을 평가합니다.")

        # 생육 데이터 경로
        file_path_biology = "./완숙토마토_생육_데이터.csv"
        data = load_and_prepare_data(file_path_biology)

        if data is not None:
            bloom_model, fruit_model = train_models(data)
            predict_with_user_input(bloom_model, fruit_model, data)

    elif page == "환경 데이터 분석":
        st.title("환경 데이터 분석")
        file_path_environment = "./토마토_환경데이터_익산_1구역(1월~11월).csv"
        data, numeric_data = load_and_process_data(file_path_environment)

        if data is not None:
            st.write("환경 데이터 미리보기:")
            st.write(data.head())

            # 계절별 평균 데이터
            seasonal_stats = numeric_data.groupby(data['계절']).agg(['mean', 'std', 'min', 'max'])
            seasonal_stats.columns = ['_'.join(col) for col in seasonal_stats.columns]
            st.write("계절별 통계 데이터")
            st.write(seasonal_stats)

if __name__ == "__main__":
    main()


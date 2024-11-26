import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# 토마토 설명 페이지
def tomato_description():
    st.title("토마토 생육 예측 시스템")
    st.subheader("1. 토마토에 대한 설명")
    st.write("""
    토마토는 전 세계에서 재배되는 대표적인 과채류입니다.  
    건강에 좋은 라이코펜과 다양한 비타민을 함유하고 있으며,  
    적정한 생육 환경과 관리가 수확량에 큰 영향을 미칩니다.  
    아래는 토마토의 이미지입니다.
    """)
    st.image("https://upload.wikimedia.org/wikipedia/commons/8/89/Tomato_je.jpg", caption="토마토")

# 생육 예측 시스템 페이지
def growth_prediction():
    st.title("토마토 생육 예측 시스템")
    st.subheader("2. 토마토 생육 예측 시스템")
    
    # 생육 데이터 예측 코드
    file_path = "완숙토마토_생육_데이터.csv"

    # 데이터 로드 및 전처리
    def load_and_prepare_data(file_path):
        try:
            data = pd.read_csv(file_path, encoding='utf-8-sig')
            columns = ['개화군', '착과군', '초장', '생장길이', '엽수', '엽장', '엽폭', '줄기굵기', '화방높이', '화방별꽃수']
            data = data[columns].dropna()  # 필요한 열만 선택하고 결측값 제거
            return data
        except FileNotFoundError:
            st.error(f"파일을 찾을 수 없습니다: {file_path}")
            return None

    # 데이터 로드
    data = load_and_prepare_data(file_path)
    if data is None:
        return

    st.write("데이터 로드 완료. 데이터 미리 보기:")
    st.write(data.head())

    # 모델 학습 및 예측
    st.header("생육 데이터 입력")
    mean_values = data[['초장', '생장길이', '엽수', '엽장', '엽폭', '줄기굵기', '화방높이', '화방별꽃수']].mean()

    user_input = {}
    for feature, example in mean_values.items():
        user_input[feature] = st.number_input(f"{feature} (예: {round(example, 2)})", value=round(example, 2))

    if st.button("예측 실행"):
        # 개화군 및 착과군 예측
        input_df = pd.DataFrame([user_input])
        bloom_model = LinearRegression()
        fruit_model = LinearRegression()

        # 모델 학습
        bloom_model.fit(data[['초장', '생장길이', '엽수', '엽장', '엽폭', '줄기굵기', '화방높이', '화방별꽃수']], data['개화군'])
        fruit_model.fit(data[['개화군']], data['착과군'])

        # 결과 예측
        bloom_count = bloom_model.predict(input_df)[0]
        fruit_count = fruit_model.predict([[bloom_count]])[0]

        # 결과 출력
        st.write(f"예측된 개화군 수: **{bloom_count:.2f}**")
        st.write(f"예측된 착과군 수: **{fruit_count:.2f}**")

# 환경 데이터 분석 페이지
def environment_analysis():
    st.title("토마토 환경 데이터 분석")
    st.subheader("3. 환경 데이터 분석")

    # 데이터 분석 함수
    def load_and_process_data(file_path):
        try:
            data = pd.read_csv(file_path, encoding='utf-8-sig')
            data['저장시각'] = pd.to_datetime(data['저장시각'], format='%Y.%m.%d %H:%M')
            data['월'] = data['저장시각'].dt.month
            data['계절'] = data['월'].apply(lambda x: '봄' if x in [3, 4, 5] else '여름' if x in [6, 7, 8] else '가을' if x in [9, 10, 11] else '겨울')
            numeric_cols = data.select_dtypes(include=['float64', 'int64']).columns
            numeric_data = data[numeric_cols]
            return data, numeric_data
        except Exception as e:
            st.error(f"데이터 처리 중 오류가 발생했습니다: {e}")
            return None, None

    # 데이터 로드
    file_path = "토마토_환경데이터_익산_1구역(1월~11월).csv"
    data, numeric_data = load_and_process_data(file_path)
    if data is None or numeric_data is None:
        return

    st.write("데이터 로드 완료. 데이터 미리 보기:")
    st.write(data.head())

    # 계절별 최적 환경 탐색
    def analyze_seasonal_conditions(numeric_data, data):
        seasonal_stats = numeric_data.groupby(data['계절']).agg(['mean', 'std', 'min', 'max'])
        seasonal_stats.columns = ['_'.join(col).strip() for col in seasonal_stats.columns.values]
        return seasonal_stats

    seasonal_stats = analyze_seasonal_conditions(numeric_data, data)
    st.write("계절별 최적 환경 데이터:")
    st.write(seasonal_stats)

# 메인 페이지
def main():
    st.sidebar.title("목차")
    page = st.sidebar.radio("이동할 페이지를 선택하세요:", ["토마토 소개", "생육 예측 시스템", "환경 데이터 분석"])

    if page == "토마토 소개":
        tomato_description()
    elif page == "생육 예측 시스템":
        growth_prediction()
    elif page == "환경 데이터 분석":
        environment_analysis()

if __name__ == "__main__":
    main()

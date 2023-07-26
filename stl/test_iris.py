import streamlit as st
from streamlit_option_menu import option_menu
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from eda import abal_data_eda, steel_data_eda


# 각 data의 path를 적으세요
abal_datapath = '../data/abalone.csv'
star_datapath = '../data/star.csv'
steel_datapath = '../data/steel.csv'

# 데이터프레임 선언
df_abal = abal_data_eda(abal_datapath)
df_star = steel_data_eda(steel_datapath)

with st.sidebar:
    choose = option_menu("판교에서 만나요", ["전복(Abalone)", "중성자별(Star)", "강판(Steel)"],
                        icons=['bi bi-droplet', 'star', 'bi bi-ticket-fill'],
                        menu_icon="bi bi-people", default_index=1)

if choose == "전복(Abalone)":
    selected1 = option_menu(None, ["데이터 해석", "머신러닝", "딥러닝"], 
    icons=['house', 'cloud-upload', "list-task", 'gear'], 
    menu_icon="cast", default_index=0, orientation="horizontal")

    if selected1 == "데이터 해석":
        # Iris 데이터셋 로드
        iris = load_iris()
        X, y = iris.data, iris.target

        # 데이터 분할
        x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # 분류 모델 생성
        model = RandomForestClassifier()
        model.fit(x_train, y_train)

        st.title('Iris 분류 예측 애플리케이션')
        st.write('사용자로부터 꽃잎과 꽃받침의 길이와 너비를 입력받아 해당하는 붓꽃의 종류를 예측하는 애플리케이션입니다.')

        # 입력 폼 구현
        sepal_length = st.slider('꽃받침 길이', 4.3, 7.9, 5.4, 0.1)
        sepal_width = st.slider('꽃받침 너비', 2.0, 4.4, 3.4, 0.1)
        petal_length = st.slider('꽃잎 길이', 1.0, 6.9, 1.3, 0.1)
        petal_width = st.slider('꽃잎 너비', 0.1, 2.5, 0.2, 0.1)

        # 입력값을 이용하여 예측
        input_data = [[sepal_length, sepal_width, petal_length, petal_width]]
        prediction = model.predict(input_data)[0]
        predicted_species = iris.target_names[prediction]

        st.write('예측 결과: ', predicted_species)


        st.title('Iris 분류 예측 애플리케이션')
        st.write('사용자로부터 꽃잎과 꽃받침의 길이와 너비를 입력받아 해당하는 붓꽃의 종류를 예측하는 애플리케이션입니다.')

        # 입력 폼 구현
        sepal_length = st.text_input('꽃받침 길이')
        sepal_width = st.text_input('꽃받침 너비')
        petal_length = st.text_input('꽃잎 길이')
        petal_width = st.text_input('꽃잎 너비')

        # 입력값을 이용하여 예측
        try:
            input_data = [[float(sepal_length), float(sepal_width), float(petal_length), float(petal_width)]]
            prediction = model.predict(input_data)[0]
            predicted_species = iris.target_names[prediction]

            st.write('예측 결과: ', predicted_species)
        except:
            st.write('유효한 숫자를 입력하세요.')
    
    if selected1 == "데이터 해석":
        st.title('새 사이드바')

        # 사이드바 생성
        st.sidebar.header('사이드바임')

        # 사이드바에 텍스트 입력 위젯 추가
        user_input = st.sidebar.text_input('아무거나 입력하셈')

        # 사이드바에 버튼 추가
        button = st.sidebar.button('눌러봐')

        # 버튼이 눌렸을 때 화면에 텍스트 표시
        if button:
            st.write(f'안녕, {user_input}!')

st.write('하하')
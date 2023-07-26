import streamlit as st
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Iris 데이터셋 로드
iris = load_iris()
X, y = iris.data, iris.target

# 데이터 분할
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 모델 선택 옵션
model_options = {
    'Random Forest': RandomForestClassifier(),
    'SVM': SVC()
}

st.title('Iris 분류 예측 애플리케이션')
st.write('사용자로부터 꽃잎과 꽃받침의 길이와 너비를 입력받아 해당하는 붓꽃의 종류를 예측하는 애플리케이션입니다.')

# 모델 선택 드롭다운
selected_model = st.selectbox('모델 선택', list(model_options.keys()))

# 선택된 모델로 예측
model = model_options[selected_model]
model.fit(x_train, y_train)

sepal_length = st.text_input('꽃받침 길이')
sepal_width = st.text_input('꽃받침 너비')
petal_length = st.text_input('꽃잎 길이')
petal_width = st.text_input('꽃잎 너비')

try:
    input_data = [[float(sepal_length), float(sepal_width), float(petal_length), float(petal_width)]]
    prediction = model.predict(input_data)[0]
    predicted_species = iris.target_names[prediction]

    st.write('예측 결과: ', predicted_species)
except:
    st.write('유효한 숫자를 입력하세요.')
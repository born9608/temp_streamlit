import time 

import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
import joblib
import keras

from streamlit_option_menu import option_menu
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from eda import abal_data_eda, steel_data_eda
from Abal_DL_local_run_v2.eval import get_Accuracy_tf

def is_float(n):
    try:
        float(n)
        return True
    except ValueError:
        return False

# 전복 딥러닝모델
path_abal = 'Abal_DL_local_run_v2/model_layers'
deep_abal_model = keras.models.load_model(path_abal, custom_objects={"get_Accuracy_tf": get_Accuracy_tf})

# 전복 머신러닝 모델
path_cat = 'model/abalone_model/final/ML_model1_pickling/cat_model_v1.pkl'
cat_model = joblib.load(path_cat)


# 전복 스케일링(ML/DL)
abal_ml_scaler = joblib.load('model/abalone_model/final/ML_model1_pickling/minmaxscaler.pkl') 
abal_dl_scaler = joblib.load('Abal_DL_local_run_v2/joblib/MinMaxScaler.joblib')

# 전복 모델 선택 옵션
abalone_model_options = {
    'Catboost': cat_model,
#    'GradientBoost': gb_model,
    'Artificial Neural Network': deep_abal_model
} 

# 별 딥러닝모델
from model.star_model.final import star_deep_model

path_lg = 'model/star_model/final/ML_model2_pickle/lg_model.pkl'
lg_model = joblib.load(path_lg)

# 별 딥러닝 파일 불러오기(ANN)
deep_model = star_deep_model.Model()
 
# 별 모델 선택 옵션
star_model_options = {
    'logistic': lg_model,
    'Artificial Neural Network': deep_model
}

# 별 스케일러(딥러닝/ 머신러닝 둘다 Standard)
star_scaler = joblib.load('model/star_model/final/ML_model2_pickle/lg_scaler.pkl')

# 강철 딥러닝모델
path_steel = 'Steel_DL_B_local_run_v1/model_layers'
deep_steel_model = keras.models.load_model(path_steel)

# 강철 머신러닝 모델
path_ml_steel = 'model/steel_model/final/ML_steel_pikling/xgb_model_multy.pkl'
ml_steel_model = joblib.load(path_ml_steel)

# 강철 스케일링(머신러닝/ 딥러닝)
steel_dl_scaler = joblib.load('Steel_DL_B_local_run_v1/joblib/StandardScaler.joblib')
steel_ml_scaler = joblib.load('model/steel_model/final/ML_steel_pikling/xgb_scaler_multy.pkl')

# 나중에 파일 정리 이후, 상대 경로 지정해 쓸 것
abal_datapath = 'data/abalone.csv'
star_datapath = 'data/star.csv'
steel_datapath = 'data/steel.csv'

encoder = LabelEncoder()

# 데이터프레임 선언
df_abal = abal_data_eda(abal_datapath)
df_star = pd.read_csv(star_datapath)
df_binary, df_multi = steel_data_eda(steel_datapath)



with st.sidebar:
    choose = option_menu("판교에서 만나요", ["Home", "전복(Abalone)", "중성자별(Star)", "강판(Steel)"],
                         icons=['bi bi-house', 'bi bi-droplet', 'star', 'bi bi-ticket-fill'],
                         menu_icon="bi bi-people", default_index=0, 
                         styles={
                        "container": {"padding": "0!important", "background-color": "#fafafa"},
                        "icon": {"color": "orange", "font-size": "18px"}, 
                        "nav-link": {"font-size": "18px", "text-align": "left", "margin":"5px", "--hover-color": "#FFE192"},
                        "nav-link-selected": {"background-coloaddr": "#FFC939"},
                               }
)

if choose == "Home":
    # 로딩바
    latest_iteration = st.empty()
    bar = st.progress(0)
    image_placeholder = st.empty()
    image_placeholder.image('image/road_to_pangyo.jpeg')

    for i in range(100):
        # 로딩바
        latest_iteration.text(f'지금 만나러 갑니다 {i+1}')
        bar.progress(i + 1)
        time.sleep(0.02)
        # 0.05 초 마다 1씩증가
    # 완료 시 로딩바 없어지면서 풍선 이펙트 보여주기 

    latest_iteration.empty()
    bar.empty()
    image_placeholder.empty()
    st.balloons()
    
    st.image('image/Pangyo_station_Exit.jpg')
    st.title("판교에서 만나요!? 팀")

    st.header("_프로젝트 소개_")

    st.markdown("##### 전복, 중성자별, 강판 경함 데이터를 기반으로 새로운 ML/DL 모델을 설계했습니다\n"
                "###### 1. 전복 모델은 성별, 무게, 크기로 고리 수를 예측하고 나이를 추론하는 모델입니다\n"
                "###### 2. 중성자별 모델은 별의 Profile과 관측치로 중성자별 여부를 판단하는 모델입니다\n"
                "###### 3. 강판 모델은 결함 검사 시 얻을 수 있는 여러 지표를 토대로 결함의 종류를 판단하는 혼합 모델입니다. "
                "경미한 결함은 이진분류 모델, 그 외의 결함은 다중분류 모델을 통해 분류합니다\n" 
                                ,unsafe_allow_html=True)
    
    tab1, tab2, tab3 = st.tabs([":sunglasses:팀원 소개", ":toolbox:웹페이지 기능", ":timer_clock:프로젝트 플랜"])

    with tab1:
        st.subheader("팀원 소개")
        st.markdown("##### 김범성<br>" ,unsafe_allow_html=True)
        st.markdown("머신러닝, streamlit 담당" ,unsafe_allow_html=True)
        st.markdown("##### 팀장 박재형<br>" ,unsafe_allow_html=True)
        st.markdown("딥러닝, 전반적인 모델성능 개선, ppt 담당" ,unsafe_allow_html=True)
        st.markdown("##### 성진현<br>" ,unsafe_allow_html=True)
        st.markdown("머신러닝, streamlit, ppt 담당" ,unsafe_allow_html=True)
        st.markdown("##### 이태령<br>" ,unsafe_allow_html=True)
        st.markdown("딥러닝, 모델성능개선, 깃허브 관리" ,unsafe_allow_html=True)

    with tab2:
        st.subheader("_이 웹페이지는_")
        st.markdown("###### 1. 모델 학습에 사용된 데이터셋의 개괄적인 정보를 제공합니다\n"
                    "###### 2. 데이터셋의 샘플 데이터와 특성 별 시각화 자료를 제공합니다\n"
                    "###### 3. 데이터 탐색으로 도출한 인사이트를 제공합니다\n"
                    "###### 4. 설계한 모델을 기반으로 간이 예측 기능을 제공합니다")
    
    with tab3:
        st.subheader("_Project progress_")
        st.image('image/project_progress.gif')
    
# 모델 학습에 사용된 데이터의 정보, 샘플, 인사이트와 모델을 통한 간이 예측 기능을 제공합니다_


if choose == "전복(Abalone)":
    selected_menu = option_menu(None, ["데이터 설명", '데이터 시각화', "모델 예측"],
                                icons=['bi bi-file-earmark', 'kanban','bi bi-gear'],
                                menu_icon="cast", default_index=0, orientation="horizontal")
    
    columns_list =  df_abal.columns.to_list()

    if selected_menu == "데이터 설명":
        selected_sub_menu = option_menu(None, ["특성 설명", "샘플 데이터", "데이터 인사이트"],
                                        menu_icon="cast", default_index=0, orientation="horizontal", 
                                        styles = {"container": {"padding": "0!important", "background-color": "#fafafa"},
                                                    "nav-link": {"font-size": "15px", "text-align": "center", "margin":"10px", "--hover-color": "#FFE192"},
                                                    "icon": {"color": "orange", "font-size": "18px"}, 
                                                    "nav-link-selected": {"background-coloaddr": "#FFC939"},
                                                })

        if selected_sub_menu == "특성 설명":
            st.header("전복 데이터")

            # 그림 두개 그리기
            abal_col1, abal_col2 = st.columns(2)
            with abal_col1:
                st.image("https://i0.wp.com/briantissot.com/wp-content/uploads/2014/09/john-exact-size100.jpg?ssl=1", use_column_width=True)
            with abal_col2:
                st.image("image/abalone.jpeg", use_column_width=True)
            
            # 특성 설명
            st.markdown('_전복 데이터셋은 4177개의 행과 9개의 특성을 갖습니다. 껍질에 나타나는 고리의 수는 전복의 나이를 책정하는 척도입니다. 성별과 크기, 무게와 관련된 8개의 특성을 기반으로 껍질의 수를 예측해야 합니다_')
            st.markdown('- **Sex** : 전복 성별 / object / F, M, I 중 하나의 값을 가진다. 유아기인 경우 I로 표시한다, 미성숙한 전복은 자웅동체이기 때문이다')
            st.markdown('- **Length** : 전복 길이 / mm / float : 껍질의 최대 길이를 의미 상단 그림 참고')
            st.markdown('- **Diameter** : 전복 지름 / mm / float : Length를 쟀을 때와 수직으로 잴 때의 길이를 말함')
            st.markdown('- **Height** : 전복 길이 / mm / float : 높이(with meat in shell)를 말함')
            st.markdown('- **Whole Weight** : 전복 전체 무게 / grams / float')
            st.markdown('- **Shucked Weight** : 껍질을 제외한 무게 / grams')
            st.markdown('- **Viscra Weight** : 내장 무게 / grams / float : 피를 빼고 측정')
            st.markdown('- **Shell Weight** : 껍질 무게 / grams / float : 건조한 뒤 껍질의 무게')
            st.markdown('- **Rings(Target)** : 전복 나이 / int : 일반적으로 껍질의 고리(Ring)수를 통해 나이를 판단한다')

        elif selected_sub_menu == "샘플 데이터":
            st.header('전복 샘플 데이터')
            with st.sidebar:   
                values = st.slider(
                    'Rings(target)에 맞춰 데이터를 확인하세요',
                    0, 30, (5, 25), step=1)
            
            df_filtered = df_abal[(df_abal['Rings'] >= values[0]) & (df_abal['Rings'] <= values[1])]
            st.dataframe(df_filtered)

        else:
            st.header('전복 데이터 인사이트')
            st.markdown("전복 데이터의 특성은 예측해야할 특성(나이)을 제외하면 8개이며 성별 특성, 크기 특성, 무게 특성 세갈래로 구분할 수 있습니다.(크기 특성: 길이, 직경, 높이 / 무게 특성: 전체 무게, 내장 무게, 껍질 무게, 껍질 제외한 무게)<br>"
                        "<br>"
                        "같은 갈래로 묶인 특성은 타겟(Rings)과 산점도를 그릴 때 매우 비슷한 분포를 보였습니다. 따라서 모델을 학습할 때 각 갈래의 대표가 될 특성을 하나씩 뽑아 데이터셋을 구성하는 시도를 했습니다, 그러나 데이터셋의 사이즈(4177X9)가 작다는 명확한 한계가 존재했기 때문에 이런 접근이 큰 효과를 발휘하지 못했습니다.<br>"
                        "<br>"
                        "전복은 유아기(Infant)에 자웅동체이며 성숙해감에 따라 성별이 고정되는 생태적 특징이 있는데 이는 데이터 분석과정에서도 확인할 수 있었습니다. 성별 특성을 원핫 인코딩 후 상관관계를 분석하면 Female, Male보다 Infant가 Rings와 더 높은 상관관계를 가집니다. 따라서 Infant인 데이터를 예측할 때 정확도가 높을 것으로 예상됩니다.<br>" 
                        "<br>"
                        "앞서 언급한대로 전복 데이터셋에 있는 특성들은 큰 틀에서 3가지로 나뉘고 그마저 크기 특성, 무게 특성은 비슷한 양태를 보입니다. 따라서 색깔, 종, 서식지 등 쉽게 측정가능하고 기록에 용이한 특성들을 추가해 데이터셋을 확장한다면 과적합 해소가 용이해질 것이고 모델 성능은 크게 개선될 것으로 사료됩니다.",
                                                        unsafe_allow_html=True)
            st.image('image/abal_venn.png')


    if selected_menu == "데이터 시각화":
        st.title('특성 별 분포 및 시각화')
        st.sidebar.header('특성 선택')

        with st.sidebar:
            option = st.selectbox('확인하고 싶은 특성을 선택하세요', columns_list)

        # 색깔 관련
        num_color = columns_list.index(option)
        sex_color = sns.color_palette("Paired")
        Sex_palette = {"F": sex_color[5], "M": sex_color[1], "I": sex_color[3]}

        if option == 'Sex':

            visual_way = ['Count Plot', 'Violin Plot']
            with st.sidebar:
                option_2nd = st.selectbox('확인하고 싶은 그래프를 선택하세요', visual_way)

            if option_2nd == 'Count Plot':

                plt.figure()
                sns.countplot(x='Sex', data=df_abal, palette=Sex_palette)
                plt.title(f'{option} Count Plot')
                st.pyplot(plt)

            else:
                plt.figure()
                sns.violinplot(x='Sex', y = "Rings", data=df_abal, palette=Sex_palette)
                plt.title(f'{option} Violin Plot')                
                st.pyplot(plt)

        else:

            visual_way = ['Kernel Distribution', 'Box Plot']
            color = sns.color_palette('husl', 10)
            with st.sidebar:
                option_2nd = st.selectbox('확인하고 싶은 그래프를 선택하세요', visual_way)

            if option_2nd == 'Kernel Distribution':
                
                plt.figure()
                sns.kdeplot(df_abal[option])
                plt.title(f'{option} Kernel Distribution')
                st.pyplot(plt)

            else:
                plt.figure()
                sns.boxplot(x=option, data=df_abal, color=color[num_color])
                plt.title(f'{option} Box plot')
                st.pyplot(plt)

    if selected_menu == "모델 예측":
        st.title('Abalone(전복 고리 예측)')
        st.write('사용자로부터 전복 데이터에 대한 입력값을 받아 고리의 수(Rings)를 예측하는 모델입니다.')

        selected_model = st.selectbox('모델 선택', list(abalone_model_options.keys()))

        # 선택된 모델로 예측
        select_model = abalone_model_options[selected_model]

        # 입력값 받기
        Sex = st.selectbox("Choose Sex(전복의 성별): ", ['F', 'M', 'I'])
        Length = st.slider("Enter Length(전복의 길이, mm): ", min_value=float(df_abal['Length'].min()), max_value=float(df_abal['Length'].max()), value=(df_abal['Length'].min()+df_abal['Length'].max())/2)
        Diameter = st.slider("Enter Diameter(전복의 직경, mm): ", min_value=float(df_abal['Diameter'].min()), max_value=float(df_abal['Diameter'].max()), value=(df_abal['Diameter'].min()+df_abal['Diameter'].max())/2)
        Height = st.slider("Enter Height(전복의 높이, mm): ", min_value=float(df_abal['Height'].min()), max_value=float(df_abal['Height'].max()), value=(df_abal['Height'].min()+df_abal['Height'].max())/2)
        Whole_Weight = st.slider("Enter Whole Weight(전복의 전체 무게, grams): ", min_value=float(df_abal['Whole weight'].min()), max_value=float(df_abal['Whole weight'].max()), value=(df_abal['Whole weight'].min()+df_abal['Whole weight'].max())/2)
        Shucked_Weigh = st.slider("Enter Shucked Weight(전복 껍질 제거 후 무게, grams): ", min_value=float(df_abal['Shucked weight'].min()), max_value=float(df_abal['Shucked weight'].max()), value=(df_abal['Shucked weight'].min()+df_abal['Shucked weight'].max())/2)
        Viscra_Weight = st.slider("Enter Viscera Weight(전복 내장 무게, grams): ", min_value=float(df_abal['Viscera weight'].min()), max_value=float(df_abal['Viscera weight'].max()), value=(df_abal['Viscera weight'].min()+df_abal['Viscera weight'].max())/2)
        Shell_Weight = st.slider("Enter Shell Weight(전복 껍질 무게, grams): ", min_value=float(df_abal['Shell weight'].min()), max_value=float(df_abal['Shell weight'].max()), value=(df_abal['Shell weight'].min()+df_abal['Shell weight'].max())/2)
        Sex_F, Sex_I, Sex_M = 0, 0, 0
    
        if Sex == 'F':
            Sex_F = 1
        
        elif Sex == 'I':
            Sex_I = 1

        elif Sex == 'M':
            Sex_M = 1   

        input_list = [Length, Diameter, Height, Whole_Weight, Shucked_Weigh, Viscra_Weight, Shell_Weight, Sex_F, Sex_I, Sex_M]

        predict_button = st.button("예측하기")

        if predict_button:
            if all([is_float(n) for n in input_list[:7]]):  # 앞의 7개 값이 실수인지 확인합니다.
                input_data_2d = np.array(input_list, dtype=float).reshape(1, -1)
                input_data_MLscaled = abal_ml_scaler.transform(input_data_2d)
                input_data_DLscaled = abal_dl_scaler.transform(input_data_2d)

                # 예측
                if select_model == cat_model:
                    prediction = select_model.predict(input_data_MLscaled)
                    # elif select_model == gb_model:
                    #     prediction = select_model.predict(input_data_MLscaled)
                else:
                    prediction = select_model.predict(input_data_DLscaled)

                st.write(f'<div style="font-size: 36px; color: blue;">예측된 고리의 수는 {prediction}</div>', unsafe_allow_html=True)

            else:
                st.write('유효한 숫자를 입력하세요.')


if choose == "중성자별(Star)":
    selected_menu = option_menu(None, ["데이터 설명", '데이터 시각화', "모델 예측"],
                                icons=['bi bi-file-earmark', 'kanban','bi bi-gear'],
                                menu_icon="cast", default_index=0, orientation="horizontal")

    if selected_menu == "데이터 설명":
        selected_sub_menu = option_menu(None, ["특성 설명", "샘플 데이터", "데이터 인사이트"],
                                        menu_icon="cast", default_index=0, orientation="horizontal", 
                                        styles = {"container": {"padding": "0!important", "background-color": "#fafafa"},
                                                    "nav-link": {"font-size": "15px", "text-align": "center", "margin":"10px", "--hover-color": "#FFE192"},
                                                    "icon": {"color": "orange", "font-size": "18px"}, 
                                                    "nav-link-selected": {"background-coloaddr": "#FFC939"},
                                                })
        if selected_sub_menu == "특성 설명":

            # 특성 설명
            st.header("중성자별 데이터")
            st.image("image/star.jpeg", use_column_width=True)
            st.markdown("_중성자별 데이터셋은 17898개의 행과 9개의 특성으로 구성됩니다. target class 특성은 0, 1로 중성자별 여부를 표현합니다. 각 별의 통합적인 Profile과 DM-SNR Curve에서 도출한 통계치로 8개 특성이 구성됩니다._")
            st.markdown('- **Mean of the Integrated Profile (통합 프로파일의 평균)**')
            st.write('통합 프로파일에서 얻은 측정값들의 평균값을 나타냅니다.')
            st.write('통합 프로파일은 여러 개의 라디오 펄스를 하나로 통합한 것으로, 천체의 특성을 반영하는 신호들이 포함되어 있습니다. 이 평균값은 천체의 펄스 특성을 나타내는 중심적인 측정치입니다.')
            st.markdown('- **Standard Deviation of the Integrated Profile (통합 프로파일의 표준 편차)**')
            st.write('통합 프로파일에서 얻은 측정값들의 편차 또는 산포를 나타냅니다. 펄스 강도가 불규칙적일수록 표준 편차가 증가합니다.')
            st.markdown('- **Excess Kurtosis of the Integrated Profile (통합 프로파일의 첨도)**')
            st.write('첨도는 확률 변수의 분포에서 꼬리 부분의 상대적인 두께를 나타내며 정규 분포에서 얼마나 벗어났는지 판단하는 지표입니다. 첨도가 0보다 크면 뾰족한 분포로, 0보다 작으면 완만한 분포로 해석할 수 있습니다.')
            st.markdown('- **Skewness of the Integrated Profile (통합 프로파일의 비대칭도)**')
            st.write('데이터의 비대칭 정도를 나타내는 지표로서, 평균을 기준으로 얼마나 좌우로 치우쳐져 있는지를 나타냅니다.양수면 오른쪽으로, 음수면 왼쪽으로 치우쳐진 분포를 의미합니다.')
            st.markdown('- **Mean of the DM-SNR Curve (DM-SNR 곡선의 평균)**')
            st.write('DM-SNR 곡선은 주기성이 있는 신호에서 찾아지는 신호 대 잡음 비율(DM-SNR) 값을 나타내는 곡선이며 이 특성은 해당 곡선의 평균값입니다')
            st.markdown('- **Standard Deviation of the DM-SNR Curve (DM-SNR 곡선의 표준 편차)**')
            st.write('DM-SNR 곡선에서 추출한 측정값들의 편차 또는 산포를 나타냅니다.')
            st.markdown('- **Excess Kurtosis of the DM-SNR Curve (DM-SNR 곡선의 첨도)**')
            st.write('DM-SNR 곡선에서 추출한 측정값들의 첨도를 나타냅니다.')
            st.markdown('- **Skewness of the DM-SNR Curve (DM-SNR 곡선의 비대칭도)**')
            st.write('DM-SNR 곡선에서 추출한 측정값들의 비대칭 정도를 나타냅니다.')
            st.markdown('- **traget_class**')
            st.write('1이면 중성자별이며 0인 경우 중성자별이 아닙니다.')
    

        elif selected_sub_menu == "샘플 데이터":
            with st.sidebar:
                star_pulsar_dict = {'Not Pulsar': 0, 'Pulsar': 1}
                star_option = st.selectbox('중성자별 여부에 따른 데이터셋을 확인하세요', options = list(star_pulsar_dict.keys()))

            st.header("중성자별 샘플 데이터")
            filtered_star_df = df_star[df_star['target_class'] == star_pulsar_dict[star_option]]
            st.dataframe(filtered_star_df)

        else:
            st.header("중성자별 데이터 인사이트")
            st.markdown("중상자별 데이터는 별의 통합 프로파일(Integrated Profile) 4개와 별의 라디오 신호를 그래픽으로 나타낸 DM-SNR Curve에서 도출한 4개의 특성으로 구성돼있습니다. 이번 프로젝트에서 8개 특성을 기반으로 중성자별 여부를 판단하는 모델을 만들었습니다<br>"
                        "<br>"
                        "중성자별은 일반적인 항성들과 매우 다른 성질을 가집니다. 첫째, 질량 대비 밀도가 매우 높습니다, 정면에서 별을 관측할 때 뒷면을 확인할 수 있을 정도로 중력이 큽니다. 둘째, 안정되지 않는 광원입니다, 방향에 따라 별의 세기가 다르고 초당 1회 이상 자전하기 때문에 관측 시 일관성이 매우 떨어집니다<br>"
                        "<br>"
                        "해당 데이터셋의 특성 별 BoxPlot을 확인하면 이상치처럼 보이는 데이터를 다수 발견할 수 있습니다, 그러나 단순 이상치라고 간주해 제거하기엔 수가 많으며 일정한 양상이 전혀 보이지 않습니다. "
                        "이상치로 보이는 데이터를 분석하면 대체로 중성자별이었으며 정상적인 분포를 보인 데이터는 대체로 중성자별이 아니었습니다. 이는 중성자별의 성질을 고려할 때 타당한 일입니다.<br>"
                        "<br>"
                        "이 데이터셋에서 일반별과 중성자별의 비율은 9:1 입니다. 그러나 두드러지게 이상한 특성값을 여러개 보유하는 중성자별의 양태를 고려할 때 업샘플링이 필수는 아닌 것으로 보입니다"
                        ,unsafe_allow_html=True)
            st.image("image/double_pulsar.gif", use_column_width=True)


    if selected_menu == "데이터 시각화":
        columns_list =  df_star.columns.to_list()
        visual_way = ['Kernel Distribution', 'Box Plot']

        with st.sidebar:
            option = st.selectbox('확인하고 싶은 특성을 선택하세요', columns_list)

        if option == 'target_class':
            option_2nd = st.selectbox('확인하고 싶은 그래프를 선택하세요', 'Count Plot')

            plt.figure()
            sns.countplot(x='target_class', data=df_star)
            plt.title(f'{option} Count Plot')
            st.pyplot(plt)
        
        else:
            with st.sidebar:
                option_2nd = st.selectbox('확인하고 싶은 그래프를 선택하세요', visual_way)  

            if option_2nd == 'Kernel Distribution':
                plt.figure()
                sns.kdeplot(df_star[option])
                plt.title(f'{option} Kernel Distribution')
                st.pyplot(plt)

            else:
                plt.figure()
                box_plot = sns.boxplot(x='target_class', y=option, data=df_star)
                box_plot.set_xticklabels(['Not Pulsar', 'Pulsar'])
                plt.title(f'Box plot of {option}, grouped by target')
                st.pyplot(plt)

    if selected_menu == '모델 예측':
        st.title('neutron star(중성자별 여부 예측)')
        st.write('사용자로부터 중성자별 데이터에 대한 입력값을 받아 중성자별을 예측하는 모델입니다.')
        # 모델 선택 드롭다운
        selected_model = st.selectbox('모델 선택', list(star_model_options.keys()))

        # 선택된 모델로 예측
        select_model = star_model_options[selected_model]

        # 입력값 받기
        # 슬라이더를 사용한 입력값 받기
        Mean_i = st.slider("Enter Mean of the integrated profile(통합 프로파일의 평균): ", min_value=float(df_star[' Mean of the integrated profile'].min()), max_value=float(df_star[' Mean of the integrated profile'].max()), value=(df_star[' Mean of the integrated profile'].min()+df_star[' Mean of the integrated profile'].max())/2)
        SD_i = st.slider("Enter Standard deviation of the integrated profile(통합 프로파일의 표준 편차): ", min_value=float(df_star[' Standard deviation of the integrated profile'].min()), max_value=float(df_star[' Standard deviation of the integrated profile'].max()), value=(df_star[' Standard deviation of the integrated profile'].min()+df_star[' Standard deviation of the integrated profile'].max())/2)
        EK_i = st.slider("Enter Excess kurtosis of the integrated profile(통합 프로파일의 첨도): ", min_value=float(df_star[' Excess kurtosis of the integrated profile'].min()), max_value=float(df_star[' Excess kurtosis of the integrated profile'].max()), value=(df_star[' Excess kurtosis of the integrated profile'].min()+df_star[' Excess kurtosis of the integrated profile'].max())/2)
        S_i = st.slider("Enter Skewness of the integrated profile(통합 프로파일의 비대칭도): ", min_value=float(df_star[' Skewness of the integrated profile'].min()), max_value=float(df_star[' Skewness of the integrated profile'].max()), value=(df_star[' Skewness of the integrated profile'].min()+df_star[' Skewness of the integrated profile'].max())/2)

        Mean_curve = st.slider("Enter Mean of the DM-SNR curve(DM-SNR 곡선의 평균): ", min_value=float(df_star[' Mean of the DM-SNR curve'].min()), max_value=float(df_star[' Mean of the DM-SNR curve'].max()), value=(df_star[' Mean of the DM-SNR curve'].min()+df_star[' Mean of the DM-SNR curve'].max())/2)
        SD_curve = st.slider("Enter Standard deviation of the DM-SNR curve(DM-SNR 곡선의 표준 편차): ", min_value=float(df_star[' Standard deviation of the DM-SNR curve'].min()), max_value=float(df_star[' Standard deviation of the DM-SNR curve'].max()), value=(df_star[' Standard deviation of the DM-SNR curve'].min()+df_star[' Standard deviation of the DM-SNR curve'].max())/2)
        EK_curve = st.slider("Enter Excess kurtosis of the DM-SNR curve(DM-SNR 곡선의 첨도): ", min_value=float(df_star[' Excess kurtosis of the DM-SNR curve'].min()), max_value=float(df_star[' Excess kurtosis of the DM-SNR curve'].max()), value=(df_star[' Excess kurtosis of the DM-SNR curve'].min()+df_star[' Excess kurtosis of the DM-SNR curve'].max())/2)
        S_curve = st.slider("Enter Skewness of the DM-SNR curve(DM-SNR 곡선의 비대칭도): ", min_value=float(df_star[' Skewness of the DM-SNR curve'].min()), max_value=float(df_star[' Skewness of the DM-SNR curve'].max()), value=(df_star[' Skewness of the DM-SNR curve'].min()+df_star[' Skewness of the DM-SNR curve'].max())/2)



        predict_button = st.button("예측하기")
        
        if predict_button:
            try:
                input_data = [[float(Mean_i), float(SD_i), float(EK_i), float(S_i), float(Mean_curve),
                            float(SD_curve), float(EK_curve), float(S_curve)]]
                
                input_data_scaled = star_scaler.transform(input_data)

                # 예측
                if select_model == lg_model:
                    prediction = select_model.predict(input_data_scaled)
                else:
                    prediction = select_model.model.predict(input_data_scaled, verbose=0)
                
                if prediction[0] == 1:
                    st.write(f'<div style="font-size: 36px; color: red;">예측 결과 : 중성자별입니다</div>', unsafe_allow_html=True)
                else:
                    st.write(f'<div style="font-size: 36px; color: green;">예측 결과 : 중성자별이 아닙니다</div>', unsafe_allow_html=True)
            except:
                st.write('유효한 숫자를 입력하세요.')


if choose == "강판(Steel)":
    selected_menu = option_menu(None, ["데이터 설명", '데이터 시각화', "모델 예측"],
                                icons=['bi bi-file-earmark', 'kanban','bi bi-gear'],
                                menu_icon="cast", default_index=0, orientation="horizontal")
    
    # 이진, 다중분류 선택해서 쓰게 하기 위해
    binary_column_list = df_binary.columns.to_list()
    multi_column_list = df_multi.columns.to_list()
    graph_list = ['kdeplot', 'histplot', 'boxplot']

    if selected_menu == "데이터 설명":
        selected_sub_menu = option_menu(None, ["특성 설명", '이진분류 데이터', "다중분류 데이터", "데이터 인사이트"],
                                        menu_icon="cast", default_index=0, orientation="horizontal", 
                                        styles = {"container": {"padding": "0!important", "background-color": "#fafafa"},
                                                    "nav-link": {"font-size": "15px", "text-align": "center", "margin":"10px", "--hover-color": "#FFE192"},
                                                    "icon": {"color": "orange", "font-size": "18px"}, 
                                                    "nav-link-selected": {"background-coloaddr": "#FFC939"},
                                                })

        if selected_sub_menu == "특성 설명":
            st.header("강판 결함 데이터")
            st.image('image/steel.jpeg')
            st.markdown('_강판 결함 데이터셋은 1942개의 행과 27개의 특성을 갖습니다. 각 특성은 결함의 기하적인 모양과 윤곽을 설명합니다. "Faulty Steel Plates"열은 6개 종류의 결함과 기타 결함으로 분류된 범주형 데이터입니다._')
            
            st.markdown('- **X_Minimum** : 결함이 있는 영역의 X 좌표 중 최소값(전처리 과정에서 Area 특성을 생성하고 제거)')
            st.markdown('- **X_Maximum** : 결함이 있는 영역의 X 좌표 중 최대값(위와 동일)')
            st.markdown('- **Y_Minimum** : 결함이 있는 영역의 Y 좌표 중 최소값(위와 동일)')
            st.markdown('- **Y_Maximum** : 결함이 있는 영역의 Y 좌표 중 최대값(위와 동일)')
            st.markdown("- **Area** : 위의 4가지를 조합해서 새로만든 컬럼")
            st.markdown("_Area(산출 공식)_ = ('X_Maximum' - 'X_Minimum') * ('Y_Maximum' - 'Y_Minimum')")
            st.markdown('- **Pixels_Areas** : 결함이 있는 영역의 픽셀 면적')
            st.markdown('- **X_Perimeter** : 결함이 있는 영역의 X 방향 둘레 길이')
            st.markdown('- **Y_Perimeter** : 결함이 있는 영역의 Y 방향 둘레 길이')
            st.markdown('- **Sum_of_Luminosity** : 결함 영역의 픽셀 밝기 합계')
            st.markdown('- **Minimum_of_Luminosity** : 결함 영역 내 최소 픽셀 밝기')
            st.markdown('- **Maximum_of_Luminosity** : 결함 영역 내 최대 픽셀 밝기')
            st.markdown('- **Length_of_Conveyer** : 컨베이어의 길이')
            st.markdown('- **TypeOfSteel_A300** : 강철 유형 A300 여부 (이진 변수)라 제거')
            st.markdown('- **TypeOfSteel_A400** : 강철 유형 A400 여부 (이진 변수)라 제거')
            st.markdown('- **TypeOfSteel** : 0은 A300, 1은 A400 (전처리를 통해 이진 변수이므로 바꿈)')
            st.markdown('- **Steel_Plate_Thickness** : 강철판 두께')
            st.markdown('- **Edges_Index** : 결함 영역 내 가장자리의 인덱스')
            st.markdown('- **Empty_Index** : 결함 영역 내 빈 공간의 인덱스')
            st.markdown('- **Square_Index** : 결함 영역이 정사각형인지를 나타내는 인덱스')
            st.markdown('- **Outside_X_Index** : 결함 영역이 X 방향 바깥쪽에 위치한 비율')
            st.markdown('- **Edges_X_Index** : 결함 영역 내 X 방향 가장자리의 인덱스')
            st.markdown('- **Edges_Y_Index** : 결함 영역 내 Y 방향 가장자리의 인덱스')
            st.markdown('- **Outside_Global_Index** : 결함 영역이 전체 영역에서 X와 Y 방향 바깥쪽에 위치한 비율 -> 특성중요도 기반 제거')
            st.markdown('- **LogOfAreas** : 결함 영역의 픽셀 면적에 대한 로그값')
            st.markdown('- **Log_X_Index** : 결함 영역의 X 좌표에 대한 로그값 -> 제거')
            st.markdown('- **Log_Y_Index** : 결함 영역의 Y 좌표에 대한 로그값 -> 제거')
            st.markdown('- **Orientation_Index** : 결함 영역의 방향 인덱스')
            st.markdown('- **Luminosity_Index** : 결함 영역의 밝기 인덱스')
            st.markdown('- **SigmoidOfAreas** : 결함 영역의 픽셀 면적에 대한 시그모이드 값')
            st.markdown('- **Type** : 강철판 결함의 종류 (다중 분류를 위한 목표 변수). 원핫인코딩 형태를 하나의 컬럼으로 정의하고 라벨인코딩 실시함')

        elif selected_sub_menu == "이진분류 데이터":
            with st.sidebar:
                bin_dict = {'기타 결함': 0, '일반 결함': 1}
                binary_option = st.selectbox('기타 결험과 일반 결함 데이터를 확인하세요', options=list(bin_dict.keys()))
                
            st.header("강판 결함 이진분류 데이터")

            filtered_binary_df = df_binary[df_binary['Type'] == bin_dict[binary_option]]
            st.dataframe(filtered_binary_df)

        elif selected_sub_menu == "다중분류 데이터":
            with st.sidebar:
                multi_option = st.selectbox('확인하고 싶은 특성을 선택하세요', df_multi['Type'].unique())
            st.header("강판 결함 다중분류 데이터")
            filtered_multi_df = df_multi[df_multi['Type'] == multi_option]
            st.dataframe(filtered_multi_df)

        else:
            st.header('강판 결함 데이터 인사이트')
            st.markdown("강판 결함 데이터는 결함의 위치, 밝기, 둘레, 넓이, 강판의 종류 등 27개의 특성으로 구성돼 있습니다. 강판 결함에 대한 많은 특성을 기반으로 결함의 종류를 판단하는 다중 분류 모델을 만들었습니다.<br>"
                        "<br>"
                        "결함의 종류 중 기타 결함(other_faults) 데이터는 결함의 양상이 일관되지 않고 복합적이었습니다. 이름이 기타 결함인만큼 주요한 결함으로 분류되지 않는 것이 다수 포함된 것으로 판단했습니다.<br>"
                        "<br>"
                        "문제는 기타 결함의 비중이 전체 데이터셋의 1/3에 달한다는 것입니다. 모든 데이터를 학습한다면 기타 결함으로 인해 다중 분류 정확도가 떨어질 수 있다고 판단했습니다. 따라서 이진분류로 기타 결함을 걸러내고 이후 다중분류에 들어가는 복합 모델을 설계했습니다<br>"
            , unsafe_allow_html=True)
            st.image('image/steel_insight.gif', use_column_width=True)

    if selected_menu == '데이터 시각화':

        st.sidebar.title('Steel Features')
        st.sidebar.write('<div style="font-size: 14px; color: black;">이중분류 데이터와 다중분류 데이터가 같이 표시됩니다.</div>', unsafe_allow_html=True)
        select_features = st.sidebar.selectbox(
                '확인하고 싶은 특성을 선택하세요', binary_column_list
            )
        select_graph = st.sidebar.selectbox(
                '확인하고 싶은 그래프를 선택하세요', graph_list
            )
        if select_graph == 'kdeplot':
            df_multi_kde = df_multi.copy()
            df_multi_kde['Type'] = encoder.fit_transform(df_multi_kde['Type'])
            # 서브플롯 생성
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))
            sns.kdeplot(data=df_binary, x=select_features, ax=ax1)
            ax1.set_title(f'Binary {select_features} Kernel')
            sns.kdeplot(data=df_multi_kde, x=select_features, ax=ax2)
            ax2.set_title(f'Multi {select_features} Kernel')
            st.pyplot(fig)
        
        elif select_graph == 'histplot':
            # 이진분류 타겟 수정
            df_binary_hist = df_binary.copy()
            df_binary_hist['Type'] = df_binary_hist['Type'].apply(lambda x: 'weak_defect' if x == 0 else 'strong_defect')
            # 서브플롯 생성
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))
            sns.histplot(data=df_binary_hist, x=select_features, ax=ax1)
            ax1.set_title(f'Binary {select_features} Kernel')
            sns.histplot(data=df_multi, x=select_features, ax=ax2)
            ax2.set_title(f'Multi {select_features} Kernel')
            st.pyplot(fig)
        else:
            if select_features == 'Type':
                st.write(f'<div style="font-size: 20px; color: red;">{select_features}은 현재 문자열이므로 {select_graph}을 확인 할 수 없습니다.</div>', unsafe_allow_html=True)
            else:
                fig = plt.figure(figsize=(15, 10))
                ax1 = fig.add_subplot(2, 1, 1)
                ax2 = fig.add_subplot(2, 1, 2)
                sns.boxplot(data=df_binary, x=select_features, ax=ax1)
                ax1.set_title(f'Binary {select_features} Kernel')
                sns.boxplot(data=df_multi, x=select_features, ax=ax2)
                ax2.set_title(f'Multi {select_features} Kernel')
                st.pyplot(fig)


    if selected_menu == '모델 예측':
        st.title('Steel(강철판 결함 예측)')
        st.write('사용자로부터 강철판 결함 데이터에 대한 입력값을 받아 결함을 예측하는 모델입니다.')
        st.write('이번 예측모델은 이진분류와 다중분류를 나누어 머신러닝/딥러닝 모델이 결합된 모델입니다.')
        st.markdown('**이진분류로 진행을 하고 이진분류에서 1(강한결함)으로 예측되면 다중분류모델에 들어가 더욱 세밀하게 예측하는 모델입니다.**')

        # 입력값 받기
        TypeOfSteel_options = {"A300": 0, "A400": 1}
        selected_TypeOfSteel  = st.selectbox("Choose TypeOfSteel(강철 유형)", options=list(TypeOfSteel_options.keys()))
        TypeOfSteel = TypeOfSteel_options[selected_TypeOfSteel]
        
        Pixels_Areas = st.slider("Enter Pixels_Areas(결함이 있는 영역의 픽셀 면적): ", min_value=float(df_binary['Pixels_Areas'].min()), max_value=float(df_binary['Pixels_Areas'].max()), value=(df_binary['Pixels_Areas'].min()+df_binary['Pixels_Areas'].max())/2)
        X_Perimeter = st.slider("Enter X_Perimeter(결함이 있는 영역의 X 방향 둘레 길이): ", min_value=float(df_binary['X_Perimeter'].min()), max_value=float(df_binary['X_Perimeter'].max()), value=(df_binary['X_Perimeter'].min()+df_binary['X_Perimeter'].max())/2)
        Y_Perimeter = st.slider("Enter Y_Perimeter(결함이 있는 영역의 Y 방향 둘레 길이): ", min_value=float(df_binary['Y_Perimeter'].min()), max_value=float(df_binary['Y_Perimeter'].max()), value=(df_binary['Y_Perimeter'].min()+df_binary['Y_Perimeter'].max())/2)
        Sum_of_Luminosity = st.slider("Enter Sum_of_Luminosity(결함 영역의 픽셀 밝기 합계): ", min_value=float(df_binary['Sum_of_Luminosity'].min()), max_value=float(df_binary['Sum_of_Luminosity'].max()), value=(df_binary['Sum_of_Luminosity'].min()+df_binary['Sum_of_Luminosity'].max())/2)
        Minimum_of_Luminosity = st.slider("Enter Minimum_of_Luminosity(결함 영역 내 최소 픽셀 밝기): ", min_value=float(df_binary['Minimum_of_Luminosity'].min()), max_value=float(df_binary['Minimum_of_Luminosity'].max()), value=(df_binary['Minimum_of_Luminosity'].min()+df_binary['Minimum_of_Luminosity'].max())/2)
        Maximum_of_Luminosity = st.slider("Enter Maximum_of_Luminosity(결함 영역 내 최대 픽셀 밝기): ", min_value=float(df_binary['Maximum_of_Luminosity'].min()), max_value=float(df_binary['Maximum_of_Luminosity'].max()), value=(df_binary['Maximum_of_Luminosity'].min()+df_binary['Maximum_of_Luminosity'].max())/2)
        Length_of_Conveyer = st.slider("Enter Length_of_Conveyer(컨베이어의 길이): ", min_value=float(df_binary['Length_of_Conveyer'].min()), max_value=float(df_binary['Length_of_Conveyer'].max()), value=(df_binary['Length_of_Conveyer'].min()+df_binary['Length_of_Conveyer'].max())/2)
        Steel_Plate_Thickness = st.slider("Enter Steel_Plate_Thickness(강철판 두께): ", min_value=float(df_binary['Steel_Plate_Thickness'].min()), max_value=float(df_binary['Steel_Plate_Thickness'].max()), value=(df_binary['Steel_Plate_Thickness'].min()+df_binary['Steel_Plate_Thickness'].max())/2)
        Edges_Index = st.slider("Enter Edges_Index(결함 영역 내 가장자리의 인덱스): ", min_value=float(df_binary['Edges_Index'].min()), max_value=float(df_binary['Edges_Index'].max()), value=(df_binary['Edges_Index'].min()+df_binary['Edges_Index'].max())/2)
        Empty_Index = st.slider("Enter Empty_Index(결함 영역 내 빈 공간의 인덱스): ", min_value=float(df_binary['Empty_Index'].min()), max_value=float(df_binary['Empty_Index'].max()), value=(df_binary['Empty_Index'].min()+df_binary['Empty_Index'].max())/2)
        Square_Index = st.slider("Enter Square_Index(결함 영역이 정사각형인지를 나타내는 인덱스): ", min_value=float(df_binary['Square_Index'].min()), max_value=float(df_binary['Square_Index'].max()), value=(df_binary['Square_Index'].min()+df_binary['Square_Index'].max())/2)
        Outside_X_Index = st.slider("Enter Outside_X_Index(결함 영역이 X 방향 바깥쪽에 위치한 비율): ", min_value=float(df_binary['Outside_X_Index'].min()), max_value=float(df_binary['Outside_X_Index'].max()), value=(df_binary['Outside_X_Index'].min()+df_binary['Outside_X_Index'].max())/2)
        Edges_X_Index = st.slider("Enter Edges_X_Index(결함 영역 내 X 방향 가장자리의 인덱스): ", min_value=float(df_binary['Edges_X_Index'].min()), max_value=float(df_binary['Edges_X_Index'].max()), value=(df_binary['Edges_X_Index'].min()+df_binary['Edges_X_Index'].max())/2)
        Edges_Y_Index = st.slider("Enter Edges_Y_Index(결함 영역 내 Y 방향 가장자리의 인덱스): ", min_value=float(df_binary['Edges_Y_Index'].min()), max_value=float(df_binary['Edges_Y_Index'].max()), value=(df_binary['Edges_Y_Index'].min()+df_binary['Edges_Y_Index'].max())/2)
        LogOfAreas = st.slider("Enter LogOfAreas(결함 영역의 픽셀 면적에 대한 로그값): ", min_value=float(df_binary['LogOfAreas'].min()), max_value=float(df_binary['LogOfAreas'].max()), value=(df_binary['LogOfAreas'].min()+df_binary['LogOfAreas'].max())/2)
        Orientation_Index = st.slider("Enter Orientation_Index(결함 영역의 방향 인덱스): ", min_value=float(df_binary['Orientation_Index'].min()), max_value=float(df_binary['Orientation_Index'].max()), value=(df_binary['Orientation_Index'].min()+df_binary['Orientation_Index'].max())/2)
        Luminosity_Index = st.slider("Enter Luminosity_Index(결함 영역의 밝기 인덱스): ", min_value=float(df_binary['Luminosity_Index'].min()), max_value=float(df_binary['Luminosity_Index'].max()), value=(df_binary['Luminosity_Index'].min()+df_binary['Luminosity_Index'].max())/2)
        SigmoidOfAreas = st.slider("Enter SigmoidOfAreas(결함 영역의 픽셀 면적에 대한 시그모이드 값): ", min_value=float(df_binary['SigmoidOfAreas'].min()), max_value=float(df_binary['SigmoidOfAreas'].max()), value=(df_binary['SigmoidOfAreas'].min()+df_binary['SigmoidOfAreas'].max())/2)
        Area = st.slider("Enter Area(결합 영역의 넓이): ", min_value=float(df_binary['Area'].min()), max_value=float(df_binary['Area'].max()), value=(df_binary['Area'].min()+df_binary['Area'].max())/2)



        predict_button = st.button("예측하기")
            
        if predict_button:
            try:
                input_data = [[float(Pixels_Areas), float(X_Perimeter), float(Y_Perimeter), float(Sum_of_Luminosity),
                                float(Minimum_of_Luminosity), float(Maximum_of_Luminosity), float(Length_of_Conveyer),
                                float(TypeOfSteel), float(Steel_Plate_Thickness), float(Edges_Index), float(Empty_Index),
                                float(Square_Index), float(Outside_X_Index), float(Edges_X_Index), float(Edges_Y_Index),
                                float(LogOfAreas), float(Orientation_Index), float(Luminosity_Index), float(SigmoidOfAreas),
                                float(Area)]]
                
                # 입력 데이터를 2차원 배열로 변환하여 스케일링
                input_data_2d = np.array(input_data).reshape(1, -1)
                
                input_data_scaled = steel_dl_scaler.transform(input_data_2d)

                # 예측
                prediction = deep_steel_model.predict(input_data_scaled, verbose=0)

                if prediction[0] == 0:
                    st.write(f'<div style="font-size: 36px; color: green;">예측 결과 : Other_Faults(약한결함) 입니다.</div>', unsafe_allow_html=True)
                else:
                    input_data_scaled_ml = steel_ml_scaler.transform(input_data_2d)

                    prediction_ml = ml_steel_model.predict(input_data_scaled_ml)
                    
                    defect_list = ['Bumps', 'Dirtiness', 'K_Scatch', 'Pastry', 'Stains', 'Z_Scratch']

                    st.write(f'<div style="font-size: 36px; color: red;">예측 결과 : {defect_list[prediction_ml[0]]} 결함입니다.</div>', unsafe_allow_html=True)
            except:
                st.write('유효한 숫자를 입력하세요.')
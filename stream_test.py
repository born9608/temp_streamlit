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
from Abal_DL_local_run_v1.EvalAccuracy import EvalAccuracy

def is_float(n):
    try:
        float(n)
        return True
    except ValueError:
        return False

# 전복 딥러닝모델
# path_abal = 'Abal_DL_local_run_v1/model_layers'
# deep_abal_model = keras.models.load_model(path_abal, custom_objects={"EvalAccuracy": EvalAccuracy})

# 전복 머신러닝 모델(Catboost, Gradientboost)
path_cat = 'model/abalone_model/final/ML_model1_pickling/cat_model_v1.pkl'
cat_model = joblib.load(path_cat)
# path_gb = 'model/abalone_model/final/ML_model1_pickling/gbm_model_v1.pkl'
# gb_model = joblib.load(path_gb)

# 전복 스케일링(ML/DL)
abal_ml_scaler = joblib.load('model/abalone_model/final/ML_model1_pickling/minmaxscaler.pkl') 
abal_dl_scaler = joblib.load('Abal_DL_local_run_v1/joblib/StandardScaler.joblib')

# 전복 모델 선택 옵션
abalone_model_options = {
    'Catboost': cat_model,
#    'GradientBoost': gb_model,
#    'Artificial Neural Network': deep_abal_model
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
    st.write("홈 화면이 구성될 페이지입니다")

if choose == "전복(Abalone)":
    selected_menu = option_menu(None, ["데이터 설명", '데이터 시각화', "모델 예측"],
                                icons=['bi bi-file-earmark', 'kanban','bi bi-gear'],
                                menu_icon="cast", default_index=0, orientation="horizontal")
    
    columns_list =  df_abal.columns.to_list()

    if selected_menu == "데이터 설명":
        selected_sub_menu = option_menu(None, ["특성 설명", "데이터 프레임 보기", "데이터 인사이트"],
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
                st.image("abalone.jpeg", use_column_width=True)
            
            # 특성 설명
            st.markdown('- **Sex** : 전복 성별 / object : F, M, I로 구성되며 I는 유아기-전복은 자웅동체이다가 성숙하면서 성별이 고정되는 경향이 있다')
            st.markdown('- **Length** : 전복 길이 / mm / float : 껍질의 최대 길이를 의미 상단 그림 참고')
            st.markdown('- **Diameter** : 전복 지름 / mm / float : Length를 쟀을 때와 수직으로 잴 때의 길이를 말함')
            st.markdown('- **Height** : 전복 길이 / mm / float : 높이(with meat in shell)를 말함')
            st.markdown('- **Whole Weight** : 전복 전체 무게 / grams / float')
            st.markdown('- **Shucked Weight** : 껍질을 제외한 무게 / grams')
            st.markdown('- **Viscra Weight** : 내장 무게 / grams / float : 피를 빼고 측정')
            st.markdown('- **Shell Weight** : 껍질 무게 / grams / float : 건조한 뒤 껍질의 무게')
            st.markdown('- **Rings(Target)** : 전복 나이 / int : 일반적으로 껍질의 고리(Ring)수를 통해 나이를 판단한다')
            st.markdown('- **특이사항**')
            st.markdown('연속성을 가지는 특성(성별과 나이를 제외한)은 원 수치에 200을 나눈 값이라 한다')
            st.markdown('EDA를 위해 임시로 200을 곱해 확인한다')

        elif selected_sub_menu == "데이터 프레임 보기":
            st.header('전복 데이터')
            with st.sidebar:   
                values = st.slider(
                    'Rings(target)에 맞춰 데이터를 확인하세요',
                    0, 30, (5, 25), step=1)
            
            df_filtered = df_abal[(df_abal['Rings'] >= values[0]) & (df_abal['Rings'] <= values[1])]
            st.dataframe(df_filtered)

        else:
            st.write("전복 데이터 인사이트가 들어갈 페이지입니다")

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
        Sex = st.selectbox("Choose Sex: ", ['F', 'M', 'I'])
        Length = st.text_input("Enter Length: ")
        Diameter = st.text_input("Enter Diameter: ")
        Height = st.text_input("Enter Height: ")
        Whole_Weight = st.text_input("Enter Whole Weight: ")
        Shucked_Weigh = st.text_input("Enter Shucked Weight: ")
        Viscra_Weight = st.text_input("Enter Viscra Weight: ")
        Shell_Weight = st.text_input("Enter Shell Weight: ")
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
                # input_data_DLscaled = abal_dl_scaler.transform(input_data_2d)

                # 예측
                if select_model == cat_model:
                    prediction = select_model.predict(input_data_MLscaled)
                    # elif select_model == gb_model:
                    #     prediction = select_model.predict(input_data_MLscaled)
                #else:
                #    prediction = select_model.predict(input_data_DLscaled)

                st.write(f'<div style="font-size: 36px; color: blue;">예측된 고리의 수는 {prediction}</div>', unsafe_allow_html=True)

            else:
                st.write('유효한 숫자를 입력하세요.')


if choose == "중성자별(Star)":
    selected_menu = option_menu(None, ["데이터 설명", '데이터 시각화', "모델 예측"],
                                icons=['bi bi-file-earmark', 'kanban','bi bi-gear'],
                                menu_icon="cast", default_index=0, orientation="horizontal")

    if selected_menu == "데이터 설명":
        selected_sub_menu = option_menu(None, ["특성 설명", "데이터 프레임 보기", "데이터 인사이트"],
                                        menu_icon="cast", default_index=0, orientation="horizontal", 
                                        styles = {"container": {"padding": "0!important", "background-color": "#fafafa"},
                                                    "nav-link": {"font-size": "15px", "text-align": "center", "margin":"10px", "--hover-color": "#FFE192"},
                                                    "icon": {"color": "orange", "font-size": "18px"}, 
                                                    "nav-link-selected": {"background-coloaddr": "#FFC939"},
                                                })
        if selected_sub_menu == "특성 설명":

            # 특성 설명
            st.header("중성자별 데이터")
            st.image("star.jpeg", use_column_width=True)
            st.markdown('- **Mean of the Integrated Profile (통합 프로파일의 평균)**')
            st.write('통합 프로파일에서 얻은 측정값들의 평균값을 나타냅니다.')
            st.write('통합 프로파일에서 얻은 측정값들의 평균값을 나타냅니다.')
            st.write('통합 프로파일은 여러 개의 라디오 펄스를 하나로 통합한 것으로, 천체의 특성을 반영하는 신호들이 포함되어 있습니다.')
            st.write('이 평균값은 천체의 펄스 특성을 나타내는 중심적인 측정치입니다.')
            st.markdown('- **Standard Deviation of the Integrated Profile (통합 프로파일의 표준 편차)**')
            st.write('통합 프로파일에서 얻은 측정값들의 편차 또는 산포를 나타냅니다.')
            st.write('평균값으로부터 얼마나 떨어져 있는지를 평가하여 데이터의 퍼짐 정도를 표현합니다.')
            st.write('표준 편차가 크면 천체의 펄스 강도가 불규칙적으로 변하는 것을 의미할 수 있습니다.')
            st.markdown('- **Excess Kurtosis of the Integrated Profile (통합 프로파일의 첨도)**')
            st.write('첨도는 확률 변수의 분포에서 꼬리 부분의 상대적인 두께를 나타내는 지표로서, 데이터가 정규 분포에서 얼마나 벗어난 분포를 갖고 있는지를 나타냅니다.')
            st.write('첨도가 0보다 크면 뾰족한 분포로, 0보다 작으면 완만한 분포로 해석할 수 있습니다.')
            st.markdown('- **Skewness of the Integrated Profile (통합 프로파일의 비대칭도)**')
            st.write('데이터의 비대칭 정도를 나타내는 지표로서, 평균을 기준으로 얼마나 좌우로 치우쳐져 있는지를 나타냅니다.')
            st.write('양수면 오른쪽으로, 음수면 왼쪽으로 치우쳐진 분포를 의미합니다.')
            st.markdown('- **Mean of the DM-SNR Curve (DM-SNR 곡선의 평균)**')
            st.write('DM-SNR 곡선은 주기성이 있는 신호에서 찾아지는 신호 대 잡음 비율(DM-SNR) 값을 나타내는 곡선입니다.')
            st.write('이 곡선에서 추출한 측정값들의 평균을 나타냅니다.')
            st.markdown('- **Standard Deviation of the DM-SNR Curve (DM-SNR 곡선의 표준 편차)**')
            st.write('DM-SNR 곡선에서 추출한 측정값들의 편차 또는 산포를 나타냅니다.')
            st.write('평균값으로부터 얼마나 떨어져 있는지를 평가하여 데이터의 퍼짐 정도를 표현합니다.')
            st.markdown('- **Excess Kurtosis of the DM-SNR Curve (DM-SNR 곡선의 첨도)**')
            st.write('DM-SNR 곡선에서 추출한 측정값들의 첨도를 나타냅니다.')
            st.write('이 값이 0보다 크면 뾰족한 분포로, 0보다 작으면 완만한 분포로 해석할 수 있습니다.')
            st.markdown('- **Skewness of the DM-SNR Curve (DM-SNR 곡선의 비대칭도)**')
            st.write('DM-SNR 곡선에서 추출한 측정값들의 비대칭 정도를 나타냅니다.')
            st.write('양수면 오른쪽으로, 음수면 왼쪽으로 치우쳐진 분포를 의미합니다.')
            st.markdown('- **traget_class**')
            st.write('1인 경우 중성자별, 0인 경우 중성자별 X')
    

        elif selected_sub_menu == "데이터 프레임 보기":
            with st.sidebar:
                star_option = st.selectbox('확인하고 싶은 타겟을 선택하세요', df_star['target_class'].unique())
            st.header("중성자별 데이터")
            st.write('0이 중성자 별이 아니고 1이 중성자 별인 데이터프레임입니다.')
            filtered_star_df = df_star[df_star['target_class'] == star_option]
            st.dataframe(filtered_star_df)

        else:
            st.write("중성자별 데이터 인사이트가 들어갈 페이지입니다")


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
        Mean_i = st.text_input("Enter Mean of the integrated profile: ")
        SD_i = st.text_input("Enter Standard deviation of the integrated profile: ")
        EK_i = st.text_input("Enter Excess kurtosis of the integrated profile: ")
        S_i = st.text_input("Enter Skewness of the integrated profile: ")
        Mean_curve = st.text_input("Enter Mean of the DM-SNR curve: ")
        SD_curve = st.text_input("Enter Standard deviation of the DM-SNR curve: ")
        EK_curve = st.text_input("Enter Excess kurtosis of the DM-SNR curve: ")
        S_curve = st.text_input("Enter Skewness of the DM-SNR curve: ")


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
            st.image('steel.jpeg')

            
            st.markdown('- **X_Minimum** : 결함이 있는 영역의 X 좌표 중 최소값 -> Area 만들고 제거')
            st.markdown('- **X_Maximum** : 결함이 있는 영역의 X 좌표 중 최대값 -> Area 만들고 제거')
            st.markdown('- **Y_Minimum** : 결함이 있는 영역의 Y 좌표 중 최소값 -> Area 만들고 제거')
            st.markdown('- **Y_Maximum** : 결함이 있는 영역의 Y 좌표 중 최대값 -> Area 만들고 제거')
            st.markdown("- **Area** : 위의 4가지를 조합해서 새로만든 컬럼")
            st.markdown('- **너비 공식**')
            st.markdown("**('X_Maximum' - 'X_Minimum') * ('Y_Maximum' - 'Y_Minimum')**")
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
            st.markdown('- **Type** : 강철판 결함의 종류 (다중 분류를 위한 목표 변수) -> 이것도 원핫인코딩 형태를 하나의 컬럼으로 정의하고 라벨인코딩 실시)')

        elif selected_sub_menu == "이진분류 데이터":
            with st.sidebar:
                binary_option = st.selectbox('확인하고 싶은 특성을 선택하세요', df_binary['Type'].unique())
            st.header("강판 결함")
            st.write('''0 -> weak_defect(약한 결함)
                    1 -> strong_defect(심한 결함)
                    ''')
            filtered_binary_df = df_binary[df_binary['Type'] == binary_option]
            st.dataframe(filtered_binary_df)

        elif selected_sub_menu == "다중분류 데이터":
            with st.sidebar:
                multi_option = st.selectbox('확인하고 싶은 특성을 선택하세요', df_multi['Type'].unique())
            st.header("강판 결함")
            filtered_multi_df = df_multi[df_multi['Type'] == multi_option]
            st.dataframe(filtered_multi_df)

        else:
            st.write('강판 결함 데이터 인사이트가 들어갈 페이지입니다')
    
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
        Pixels_Areas = st.text_input("Enter Pixels_Areas: ")
        X_Perimeter = st.text_input("Enter X_Perimeter: ")
        Y_Perimeter = st.text_input("Enter Y_Perimeter: ")
        Sum_of_Luminosity = st.text_input("Enter Sum_of_Luminosity: ")
        Minimum_of_Luminosity = st.text_input("Enter Minimum_of_Luminosity: ")
        Maximum_of_Luminosity = st.text_input("Enter Maximum_of_Luminosity: ")
        Length_of_Conveyer = st.text_input("Enter Length_of_Conveyer: ")
        TypeOfSteel = st.selectbox("Choose TypeOfSteel", options=[0, 1])
        Steel_Plate_Thickness = st.text_input("Enter Steel_Plate_Thickness: ")
        Edges_Index = st.text_input("Enter Edges_Index: ")
        Empty_Index = st.text_input("Enter Empty_Index: ")
        Square_Index = st.text_input("Enter Square_Index: ")
        Outside_X_Index = st.text_input("Enter Outside_X_Index: ")
        Edges_X_Index = st.text_input("Enter Edges_X_Index: ")
        Edges_Y_Index = st.text_input("Enter Edges_Y_Index: ")
        LogOfAreas = st.text_input("Enter LogOfAreas: ")
        Orientation_Index = st.text_input("Enter Orientation_Index: ")
        Luminosity_Index = st.text_input("Enter Luminosity_Index: ")
        SigmoidOfAreas = st.text_input("Enter SigmoidOfAreas: ")
        Area = st.text_input("Enter Area: ")


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
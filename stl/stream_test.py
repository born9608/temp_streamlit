import streamlit as st
from streamlit_option_menu import option_menu
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder
import seaborn as sns
from eda import abal_data_eda, steel_data_eda

# 나중에 파일 정리 이후 상대 경로 지정해 쓸 것
abal_datapath = 'data/abalone.csv'
star_datapath = 'data/star.csv'
steel_datapath = 'data/steel.csv'


encoder = LabelEncoder()

# 데이터프레임 선언
df_abal = abal_data_eda(abal_datapath)
df_binary, df_multi = steel_data_eda(steel_datapath)

with st.sidebar:
    choose = option_menu("판교에서 만나요", ["전복(Abalone)", "중성자별(Star)", "강판(Steel)"],
                         icons=['bi bi-droplet', 'star', 'bi bi-ticket-fill'],
                         menu_icon="bi bi-people", default_index=1)

if choose == "전복(Abalone)":
    selected_menu = option_menu(None, ["데이터 설명", 'EDA', "모델 예측"],
    icons=['bi bi-file-earmark', 'kanban','bi bi-gear'],
    menu_icon="cast", default_index=0, orientation="horizontal")
    
    columns_list =  df_abal.columns.to_list()

    if selected_menu == "데이터 설명":
        pass

    if selected_menu == "EDA":
        st.title('특성 별 분포 및 시각화')
        st.sidebar.header('특성 선택')

        with st.sidebar:
            option = st.selectbox('특성을 골라주세요', columns_list)

        # 색깔 관련
        num_color = columns_list.index(option)
        sex_color = sns.color_palette("Paired")
        Sex_palette = {"F": sex_color[5], "M": sex_color[1], "I": sex_color[3]}

        if option == 'Sex':

            visual_way = ['Count Plot', 'Violin Plot']
            with st.sidebar:
                option_2nd = st.selectbox('시각화 방법을 선택하세요', visual_way)

            if option_2nd == 'Count Plot':

                plt.figure()
                sns.countplot(x='Sex', data=df_abal, palette=Sex_palette)
                st.pyplot(plt)

            else:
                plt.figure()
                sns.violinplot(x='Sex', y = "Rings", data=df_abal, palette=Sex_palette)
                st.pyplot(plt)

        else:

            visual_way = ['Kernel Distribution', 'Box Plot']
            color = sns.color_palette('husl', 10)
            with st.sidebar:
                option_2nd = st.selectbox('시각화 방법을 선택하세요', visual_way)

            if option_2nd == 'Kernel Distribution':
                
                plt.figure()
                sns.kdeplot(df_abal[option])
                plt.title(f'{option} Kernel Distribution')
                st.pyplot(plt)

            else:
                plt.figure()
                sns.boxplot(x=option, data=df_abal, color=color[num_color])
                plt.title(f'{option} Box Plot')
                st.pyplot(plt)

if choose == "강판(Steel)":
    selected_menu = option_menu(None, ["데이터 설명", 'EDA', "모델 예측"],
    icons=['bi bi-file-earmark', 'kanban','bi bi-gear'],
    menu_icon="cast", default_index=0, orientation="horizontal")
    
    # 이진, 다중분류 선택해서 쓰게 하기 위해
    binary_column_list = df_binary.columns.to_list()
    multi_column_list = df_multi.columns.to_list()
    graph_list = ['kdeplot', 'histplot']
    if selected_menu == "데이터 설명":
        pass
    if selected_menu == 'EDA':
        
        st.sidebar.title('Steel Features')
        st.sidebar.write('<div style="font-size: 14px; color: black;">이중분류 데이터와 다중분류 데이터가 같이 표시됩니다.</div>', unsafe_allow_html=True)
        select_features = st.sidebar.selectbox(
                '확인하고 싶은 특성을 선택하세요', binary_column_list
            )
        select_graph = st.sidebar.selectbox(
                '확인하고 싶은 그래프를 선택하세요', graph_list
            )
        if st.sidebar.button("확인"):
            if select_graph == 'kdeplot':
                df_multi_kde = df_multi.copy()
                df_multi_kde['Type'] = encoder.fit_transform(df_multi_kde['Type'])
                # 서브플롯 생성
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
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
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
                sns.histplot(data=df_binary_hist, x=select_features, ax=ax1)
                ax1.set_title(f'Binary {select_features} Kernel')
                sns.histplot(data=df_multi, x=select_features, ax=ax2)
                ax2.set_title(f'Multi {select_features} Kernel')
                st.pyplot(fig)

    if selected_menu == '모델 예측':
        pass
                
    else:
        pass



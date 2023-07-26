import streamlit as st
from streamlit_option_menu import option_menu
import pandas as pd
import matplotlib.pyplot as plt

import seaborn as sns
from eda import abal_data_eda, steel_data_eda

abal_datapath = 'data/abalone.csv'
star_datapath = 'data/star.csv'
steel_datapath = 'data/steel.csv'

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


                
    else:
        pass



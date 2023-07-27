import pandas as pd

def abal_data_eda(df_path):
    # original 데이터셋 불러오기
    df = pd.read_csv(df_path)

    # 이상치 전처리 
    df_origin = df.copy()
    df_origin.iloc[:, 1:8] *= 200

    selected_rows = df_origin[df_origin['Diameter'] == 46.0]
    selected_rows_removed = selected_rows.drop(3996)
    df_origin.loc[3996, 'Height'] = selected_rows_removed['Height'].mean()

    selected_rows = df_origin[df_origin['Diameter'] == 71.0]
    selected_rows_removed = selected_rows.drop(2051)
    df_origin.loc[2051, 'Height'] = selected_rows_removed['Height'].mean()

    selected_rows = df_origin[df_origin['Diameter'] == 68.0]
    selected_rows_removed = selected_rows.drop(1257)
    df_origin.loc[1257, 'Height'] = selected_rows_removed['Height'].mean()

    selected_rows = df_origin[df_origin['Length'] == 141.0]
    selected_rows_removed = selected_rows.drop(1417)
    df_origin.loc[1417, 'Height'] = selected_rows_removed['Height'].mean()

    condition = (df_origin['Whole weight'] > 12.0) & (df_origin['Whole weight'] < 14.0)
    selected_rows = df_origin[condition]
    selected_rows_removed = selected_rows.drop(3522)
    df_origin.loc[3522, 'Height'] = selected_rows_removed['Height'].mean()

    df_clean = df_origin.copy()
    df_clean.iloc[:, 1:8] /= 200

    return df_clean


def steel_data_eda(df_path):
    
    df = pd.read_csv(df_path)
    # 이상치 제거
    df = df.drop(df.nlargest(1, 'Pixels_Areas').index)
    # 이상치 제거 후 reset_index
    df = df.reset_index(drop = True)
    # sparse한 Target 컬럼 7개를 Type 특성을 만들어 하나로 뭉치고 기존 Target 컬럼을 제거한다
    target_list = ['Pastry', 'Z_Scratch', 'K_Scatch', 'Stains', 'Dirtiness', 'Bumps', 'Other_Faults']
    df['Type'] = df[target_list].idxmax(axis=1)
    df = df.drop(columns=target_list)
    # Area라는 컬럼 생성
    df['Area'] = (df['X_Maximum'] - df['X_Minimum']) * (df['Y_Maximum'] - df['Y_Minimum'])
    # Area컬럼을 생성하는데 사용된 피쳐 제거(X_Maximum, X_Minimum, Y_Maximum, Y_Minimum)
    df = df.drop(df.iloc[:, :4].columns, axis=1)
    # A300 제거(300 or 400이라 하나 제거 후 컬럼명 어떻게 할지 고민)
    df = df.drop('TypeOfSteel_A300', axis=1)
    # 일단 TypeOfSteel로만 변경해서 설명란에 0은 300 1은 400으로 해보기로
    df.rename(columns={'TypeOfSteel_A400':'TypeOfSteel'}, inplace=True)
    # Log_X_index, Log_Y_index 제거 LogOFAreas가 합친결과값으로 판단되어 제거하기로 함
    df = df.drop(['Log_X_Index', 'Log_Y_Index'], axis=1)
    df = df.drop('Outside_Global_Index', axis=1)

    # 이진분류 타겟수정
    df_clear1 = df.copy()
    df_clear1['Type'] = df_clear1['Type'].apply(lambda x: 1 if x != 'Other_Faults' else 0)

    # 다중분류 타겟수정
    df_clear2 = df.copy()
    # 'Other_Faults'를 제외한 인덱스 가져오기
    indices_to_remove = df_clear2[df_clear2['Type'] == 'Other_Faults'].index
    df_clear2 = df_clear2.drop(indices_to_remove)
    
    return df_clear1, df_clear2
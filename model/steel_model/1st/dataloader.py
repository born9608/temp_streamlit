import pandas as pd

def load_data(path = '../../../data/steel.csv', help = False):

    if help == True:
        
        print("""    
        load_data함수는 원본 데이터를 가져와 다음과 같이 전처리한다

        1. 하나의 이상치를 제거한다
        2. sparse한 Target 컬럼 7개를 제거하고 하나의 특성(Type)으로 만든다 
        3. Area 특성을 만들고 다음 4개의 특성(X_Maximum, X_Minimum, Y_Maximum, Y_Minimum)을 제거한다
        4. A300을 0, A400을 1로 하는 TypeOfSteel 특성을 생성하고 A300, A400은 제거한다
        5. Log_X_index, Log_Y_index를 제거한다 
        6. Outside_Global_Index를 제거한다 
              
        원본 데이터 기준 1940개의 행과 23개의 열(22개의 특성 + 1개의 타겟: Type)을 갖는다
        """)
    
    df = pd.read_csv(path)

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

    target_list = ['Pastry', 'Z_Scratch', 'K_Scatch', 'Stains', 'Dirtiness', 'Bumps', 'Other_Faults']

    # Area컬럼을 생성하는데 사용된 피쳐 제거(X_Maximum, X_Minimum, Y_Maximum, Y_Minimum)
    df = df.drop(df.iloc[:, :4].columns, axis=1)

    # A300 제거(300 or 400이라 하나 제거 후 컬럼명 어떻게 할지 고민)
    df = df.drop('TypeOfSteel_A300', axis=1)
    # 일단 TypeOfSteel로만 변경해서 설명란에 0은 300 1은 400으로 해보기로
    df.rename(columns={'TypeOfSteel_A400':'TypeOfSteel'}, inplace=True)

    # Log_X_index, Log_Y_index 제거 LogOFAreas가 합친결과값으로 판단되어 제거하기로 함
    df = df.drop(['Log_X_Index', 'Log_Y_Index'], axis=1)
    df = df.drop('Outside_Global_Index', axis=1)

    return df
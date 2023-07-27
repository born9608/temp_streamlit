import joblib
import numpy as np
import warnings
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, Normalizer, RobustScaler, MaxAbsScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import ADASYN
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.utils import get_custom_objects
from tensorflow.keras.regularizers import L1L2
from timeit import default_timer as timer
import json
from EvalAccuracy import EvalAccuracy


warnings.filterwarnings("ignore", category=UserWarning)

def predict_abalone_rings():
    # load_model()로 모델 불러오기 -> custom_objects 버그 해결 안됨, 평가값이 달라짐
    model = keras.models.load_model('model_layers', custom_objects={"EvalAccuracy": EvalAccuracy })








#    # load_weights()로 모델 구현 후 load_weights로 가중치만 불러오기 -> 코드가 길어짐
#    def get_model(shape=len(X_train.columns), dout=0.01):
#        regularizer=keras.regularizers.L1L2(l1=0.001, l2=0.001)
#
#        inputs  = keras.Input(shape=shape)
#
#        dense1  = keras.layers.Dense(12, kernel_regularizer=regularizer)(inputs)
#        norm1   = keras.layers.BatchNormalization()(dense1)
#        relu1   = keras.layers.Activation('relu')(norm1)
#        dout1   = keras.layers.Dropout(dout)(relu1)
#
#        dense2  = keras.layers.Dense(10, kernel_regularizer=regularizer)(dout1)
#        norm2   = keras.layers.BatchNormalization()(dense2)
#        relu2   = keras.layers.Activation('relu')(norm2)
#        dout2   = keras.layers.Dropout(dout)(relu2)
#
#        dense3  = keras.layers.Dense(8, kernel_regularizer=regularizer)(dout2)
#        norm3   = keras.layers.BatchNormalization()(dense3)
#        relu3   = keras.layers.Activation('relu')(norm3)
#        dout3   = keras.layers.Dropout(dout)(relu3)
#
#        dense4  = keras.layers.Dense(6, kernel_regularizer=regularizer)(dout3)
#        norm4   = keras.layers.BatchNormalization()(dense4)
#        relu4   = keras.layers.Activation('relu')(norm4)
#        dout4   = keras.layers.Dropout(dout)(relu4)
#
#        dense5  = keras.layers.Dense(4, kernel_regularizer=regularizer)(dout4)
#        norm5   = keras.layers.BatchNormalization()(dense5)
#        relu5   = keras.layers.Activation('relu')(norm5)
#        dout5   = keras.layers.Dropout(dout)(relu5)
#
#        concat1  = keras.layers.Concatenate(axis=1)([dout5, dout4])
#        dense6  = keras.layers.Dense(6, kernel_regularizer=regularizer)(concat1)
#        norm6   = keras.layers.BatchNormalization()(dense6)
#        relu6   = keras.layers.Activation('relu')(norm6)
#        dout6   = keras.layers.Dropout(dout)(relu6)
#
#        concat2  = keras.layers.Concatenate(axis=1)([dout6, dout3])
#        dense7  = keras.layers.Dense(8, kernel_regularizer=regularizer)(concat2)
#        norm7   = keras.layers.BatchNormalization()(dense7)
#        relu7   = keras.layers.Activation('relu')(norm7)
#        dout7   = keras.layers.Dropout(dout)(relu7)
#
#        concat3  = keras.layers.Concatenate(axis=1)([dout7, dout2])
#        dense8  = keras.layers.Dense(10, kernel_regularizer=regularizer)(concat3)
#        norm8   = keras.layers.BatchNormalization()(dense8)
#        relu8   = keras.layers.Activation('relu')(norm8)
#        dout8   = keras.layers.Dropout(dout)(relu8)
#
#        concat4  = keras.layers.Concatenate(axis=1)([dout8, dout1])
#        dense9  = keras.layers.Dense(12, kernel_regularizer=regularizer)(concat4)
#        norm9   = keras.layers.BatchNormalization()(dense9)
#        relu9   = keras.layers.Activation('relu')(norm9)
#        dout9   = keras.layers.Dropout(dout)(relu9)
#
#        outputs = keras.layers.Dense(1)(dout9)
#        model   = keras.Model(inputs, outputs, name='Abalone_Model')
#
#        model.compile(
#            optimizer=keras.optimizers.Adam(),
#            loss=keras.losses.MeanSquaredError(),
#            metrics=[EvalAccuracy()]
#            )
#    return model
#
#    model = get_model()






    # 2. scaler 불러오기
    scaler = joblib.load('joblib/MinMaxScaler.joblib')
    
    # 입력값 받기
    Sex = input("Enter Sex(F, M, I): ")
    Length = input("Enter Length: ")
    Diameter = input("Enter Diameter: ")
    Height = input("Enter Height: ")
    Whole_weight = input("Enter Whole weight: ")
    Shucked_weight = input("Enter Shucked weight: ")
    Viscera_weight = input("Enter Viscera weight: ")
    Shell_weight = input("Enter Shell weight: ")
    Rings = input("Enter Real Rings to compare with Prediction(Target): ")
    Sex_F, Sex_I, Sex_M = 0, 0, 0
    
    if Sex == 'F':
        Sex_F = 1
    
    if Sex == 'I':
        Sex_I = 1

    if Sex == 'M':
        Sex_M = 1


    # 리스트로 변형
    input_list = [Length, Diameter, Height, Whole_weight, Shucked_weight, Viscera_weight, Shell_weight, Sex_F, Sex_I, Sex_M]
    
    # 입력 데이터를 2차원 배열로 변환하여 스케일링
    input_data_2d = np.array(input_list).reshape(1, -1)
    input_data_scaled = scaler.transform(input_data_2d)

    # 예측
    prediction = model.predict(input_data_scaled)
    print(prediction, Rings)

predict_abalone_rings()

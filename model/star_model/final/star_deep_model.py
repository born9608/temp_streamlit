import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import ADASYN

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers as Layer
from tensorflow.keras.metrics import Recall, Precision, BinaryAccuracy, TruePositives, TrueNegatives, FalsePositives, FalseNegatives
from tensorflow.keras.regularizers import L1L2
from timeit import default_timer as timer

class Model():
    def __init__(self, weight='model/star_model/final/star_weight.h5'):
      
       # 모델 생성
       self.model = self.modelling()
       self.model.load_weights(weight)
  
    def modelling(self, drop_rate= 0.1, activation= 'relu'):
       # 모델 만들기
       # 뉴런의 개수는 input과 output 사이의 숫자를 넣으라고 한다. --> 사용, 왜냐하면 시간이 적게 걸려서
       # 뉴런의 개수는 input의 2/3 정도 넣라고 한다.
       # 뉴런의 개수는 input의 두배보다는 적게 넣라고 한다.
       self.regularizer = L1L2(l1=0.001, l2=0.001)
       
       model = Sequential([Layer.Dense(12, input_shape=(8,), name='input_layer')])
       model.add(Layer.Dense(8, activation = activation, kernel_regularizer=self.regularizer, name='dense_input'))
       model.add(Layer.BatchNormalization(name='batch_norm_1'))
       model.add(Layer.Dropout(drop_rate, name='dropout_1'))

       model.add(Layer.Dense(6, activation = activation, kernel_regularizer=self.regularizer, name='dense_2'))
       model.add(Layer.BatchNormalization(name='batch_norm_2'))
       model.add(Layer.Dropout(drop_rate, name='dropout_2'))
       
       model.add(Layer.Dense(4, activation = activation, kernel_regularizer=self.regularizer, name='dense_3'))
       model.add(Layer.BatchNormalization(name='batch_norm_3'))
       model.add(Layer.Dropout(drop_rate, name='dropout_3'))
       model.add(Layer.Dense(1, activation = 'sigmoid', name='output'))
       
       # metrics에는 1이 나오는 recall 이랑 acc만 중요하다.
       # f1은 cm을 통해서 알 수 있고 라이브러리가 존재하지 않기에 안 넣음.
       metrics = [
          Recall(name = 'recall'),
          Precision(name = 'precision'),
          BinaryAccuracy(name = 'binary accuracy') # Accuracy를 사용 안 하는 이유는 Accuracy가 이상하세 나왔기 때문.
          ]
       
       model.compile(optimizer = 'adam',
                     loss='binary_crossentropy',
                     metrics = metrics)
       
       return model
    
   #  def own_input(self):
   #      Mean_i = input("Enter Mean of the integrated profile: ")
   #      SD_i = input("Enter Standard deviation of the integrated profile: ")
   #      EK_i = input("Enter Excess kurtosis of the integrated profile: ")
   #      S_i = input("Enter Skewness of the integrated profile: ")
   #      Mean_curve = input("Enter Mean of the DM-SNR curve: ")
   #      SD_curve = input("Enter Standard deviation of the DM-SNR curve: ")
   #      EK_curve = input("Enter Excess kurtosis of the DM-SNR curve: ")
   #      S_curve = input("Enter Skewness of the DM-SNR curve: ")
   #      input_list = [Mean_i, SD_i, EK_i, S_i, Mean_curve,SD_curve, EK_curve, S_curve]
   #      print(f'input list: {input_list}')

   #      input_scaled = self.scaler.transform([input_list])

   #      result = self.model.predict(input_scaled, verbose = 0)

   #      if result[0] == 1:
   #          return '중성자별입니다'
   #      else:
   #          return '중성자별이 아닙니다'
    
# result = Model().own_input()

# # 결과 출력
# print("Result:", result)
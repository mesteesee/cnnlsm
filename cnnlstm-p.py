import pandas as pd
import numpy as np
from tensorflow import keras
import matplotlib.pyplot as plt
import tensorflow as tf
import h5py

# 讀取資料
predict_data = pd.read_csv('predict.csv', encoding="UTF-8")
encoder = pd.read_csv('encoder.csv', encoding="UTF-8")

# 載入先前訓練好的模型
model = keras.models.load_model('cnnlstm.h5')

# 資料預處理
def preprocess_data(encoder):
    graph_encoder = {row['圖號1']: i for i, row in encoder.iterrows()}
    graph_encoder2 = {row['圖號2']: i for i, row in encoder.iterrows()}
    encoder = encoder.drop(columns=['圖號1','圖號2'])

    # 建立類別到編碼的映射字典
    category_maps = {}
    for col in encoder.columns:
        category_maps[col] = {row[col]: i for i, row in encoder.iterrows()}
        
    return graph_encoder, graph_encoder2, category_maps

# 呼叫資料預處理函數
graph_encoder, graph_encoder2, category_maps = preprocess_data(encoder)

X_predict = predict_data[['蓋材', '罐材', '罐型','蓋厚度', '罐厚度']]

def predict_cnnlstm(model, input_data):

    # 編碼類別特徵
    cat_cols = ['蓋材', '罐材',  '罐型'] 
    for i, col in enumerate(cat_cols): 
        input_data[i] = category_maps[col][input_data[i]]  

    # 不進行數值特徵縮放
    num_input = [input_data[3], input_data[4]] 

    # 構造模型輸入  
    input_array = np.array(input_data[:3] + num_input)
    input_array = input_array.reshape(1, 1, -1)

    # 進行預測
    prediction = model.predict(input_array)
    predicted_class_1 = np.argmax(prediction[0])  # 第一個輸出層的預測
    predicted_class_2 = np.argmax(prediction[1])  # 第二個輸出層的預測

    # 將預測值轉換回原始圖號
    original_prediction_1 = list(graph_encoder.keys())[list(graph_encoder.values()).index(predicted_class_1)]
    original_prediction_2 = list(graph_encoder2.keys())[list(graph_encoder2.values()).index(predicted_class_2)]

    return original_prediction_1, original_prediction_2 ,predicted_class_1 ,predicted_class_2

# 創建一個空列表來存儲預測結果
all_predictions = []

# 使用 apply 遍歷 DataFrame 中的每一行
for index, row in X_predict.iterrows():
    # 將行數據轉換為列表
    input_data_list = row.tolist()

    # 使用 predict_cnnlstm 函數進行預測
    predicted_class_1, predicted_class_2, original_prediction_1, original_prediction_2 = predict_cnnlstm(model, input_data_list)

    # 將預測結果添加到列表中
    all_predictions.append((predicted_class_1, predicted_class_2, original_prediction_1, original_prediction_2))

for prediction in all_predictions:
    print(prediction)



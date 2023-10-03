import pandas as pd
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt


# 讀取資料
train_data = pd.read_parquet('train.parquet', engine='pyarrow')
val_data = pd.read_parquet('val.parquet', engine='pyarrow')
test_data= pd.read_parquet('test.parquet', engine='pyarrow')
encoder = pd.read_parquet('encoder.parquet', engine='pyarrow')

# 資料預處理
def preprocess_data(train_data, val_data, test_data, encoder):
    graph_encoder = {row['1號溝型']: i for i, row in encoder.iterrows()}
    graph_encoder2 = {row['2號溝型']: i for i, row in encoder.iterrows()}

    encoder = encoder.drop(columns=['1號溝型','2號溝型'])

    # 建立類別到編碼的映射字典
    category_maps = {}
    for col in encoder.columns:
        category_maps[col] = {row[col]: i for i, row in encoder.iterrows()}

    # 使用編碼字典對類別特徵進行編碼
    for col in category_maps:
        train_data[col] = train_data[col].map(category_maps[col])
        val_data[col] = val_data[col].map(category_maps[col])
        test_data[col] = test_data[col].map(category_maps[col])

    return train_data, val_data, test_data, graph_encoder, graph_encoder2, category_maps

# 呼叫資料預處理函數
train_data, val_data, test_data, graph_encoder, graph_encoder2, category_maps = preprocess_data(train_data, val_data, test_data, encoder)

# 提取特徵和標籤
X_train = train_data[['蓋材', '罐材', '罐型','蓋厚度', '罐厚度']]
y_train = [train_data['1號溝型'], train_data['2號溝型']]
X_val = val_data[['蓋材', '罐材', '罐型','蓋厚度', '罐厚度']]
y_val = [val_data['1號溝型'], val_data['2號溝型']]
X_test = test_data[['蓋材', '罐材', '罐型','蓋厚度', '罐厚度']]
y_test = [test_data['1號溝型'], test_data['2號溝型']]


# 圖號編碼
y_train[0] = y_train[0].map(graph_encoder)
y_train[1] = y_train[1].map(graph_encoder2) 
y_val[0] = y_val[0].map(graph_encoder)
y_val[1] = y_val[1].map(graph_encoder2)
y_test[0] = y_test[0].map(graph_encoder)
y_test[1] = y_test[1].map(graph_encoder2)

num_classes1 = len(y_train[0].unique())
num_classes2 = len(y_train[1].unique())

X_train = X_train.values[:, np.newaxis, :]
X_val = X_val.values[:, np.newaxis, :]
X_test = X_test.values[:, np.newaxis, :]


# 建立多輸出模型
def build_multi_output_model(input_shape, num_classes1, num_classes2):
    shared_input = layers.Input(shape=input_shape)
    x = layers.LSTM(64, return_sequences=True)(shared_input)
    x = layers.LSTM(64, return_sequences=True)(x)
    x = layers.Dropout(0.25)(x)
    x = layers.TimeDistributed(layers.Dense(20))(x)
    x = layers.Flatten()(x)
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dropout(0.25)(x)
    x = layers.Dense(64, activation='relu')(x)
    
    # 第一个输出层
    output1 = layers.Dense(num_classes1, activation='softmax', name='output1')(x)
    
    # 第二个输出层
    output2 = layers.Dense(num_classes2, activation='softmax', name='output2')(x)
    
    model = keras.Model(inputs=shared_input, outputs=[output1, output2])
    
    return model

# 建立模型
input_shape = X_train.shape[1:]
model = build_multi_output_model(input_shape, num_classes1, num_classes2)
# 定義回調以提前停止訓練
early_stopping = EarlyStopping(monitor='val_loss', patience=15)

# 編譯模型
model.compile(
  optimizer=keras.optimizers.Adam(learning_rate=0.001),
  loss={'output1': 'sparse_categorical_crossentropy', 'output2': 'sparse_categorical_crossentropy'},
  metrics={'output1': 'accuracy', 'output2': 'accuracy'},
)



# 訓練模型
epochs = 1000
batch_size = 32

history = model.fit(
  X_train, {'output1': y_train[0], 'output2': y_train[1]},
  epochs=epochs, 
  batch_size=batch_size,
  validation_data=(X_val, {'output1': y_val[0], 'output2': y_val[1]}),
  callbacks=[early_stopping]
)

# 評估模型
test_results = model.evaluate(X_test, {'output1': y_test[0], 'output2': y_test[1]})

# 獲取損失和準確度
test_loss = round(test_results[0],2)
test_acc_output1 = round(test_results[3],2)
test_acc_output2 = round(test_results[4],2)

# 印出測試損失和準確度
print("Test Loss:", test_loss)
print("Test Accuracy (1#):", test_acc_output1)
print("Test Accuracy (2#):", test_acc_output2)


# 畫訓練曲線
plt.subplot(1, 3, 1)
plt.plot(history.history['loss']) 
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Val'], loc='upper left')
plt.subplot(1, 3, 2)
plt.plot(history.history['output1_accuracy']) 
plt.plot(history.history['val_output1_accuracy'])
plt.title('Model output1_accuracy')
plt.ylabel('output1_accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Val'], loc='upper left')
plt.subplot(1, 3, 3)
plt.plot(history.history['output2_accuracy']) 
plt.plot(history.history['val_output2_accuracy'])
plt.title('Model output2_accuracy')
plt.ylabel('output2_accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Val'], loc='upper left')

plt.tight_layout()
plt.show()

# 儲存模型
model.save('cnnlstm.h5')

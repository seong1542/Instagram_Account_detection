# -*- coding: utf-8 -*-
"""Instagram_DeepLearning.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1SXnUV7Y0ZwcjoFC107GSweDcAVLaDbqo
"""

import numpy as np
import pandas as pd
from google.colab import drive
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import ModelCheckpoint
drive.mount('/content/gdrive',force_remount=True)

filepath='/content/gdrive/My Drive/'
train = pd.read_csv(filepath+'instagram_train.csv')
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
scaler.fit(train)
train = pd.DataFrame(data = scaler.transform(train), columns = train.columns, index = train.index)
y_train = train['#fake']
X_train = train.drop('#fake',axis=1)
from sklearn.model_selection import train_test_split
X_train,X_valid, y_train, y_valid = train_test_split(X_train,y_train, test_size=0.2, random_state=0, shuffle=True)

print('X_train = ', X_train.shape)
print('X_valid = ', X_valid.shape)
print('y_train = ', y_train.shape)
print('y_valid = ', y_valid.shape)
y_train = tf.keras.utils.to_categorical(y_train, num_classes=2)
y_valid = tf.keras.utils.to_categorical(y_valid, num_classes=2)
print(y_train.shape, y_valid.shape)

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, BatchNormalization, Dropout

def dnn_model():
  model=Sequential()

  #hidden layer #1
  model.add(Dense(units = (256), input_dim = 16))
  model.add(BatchNormalization())
  model.add(Activation('relu'))
  model.add(Dropout(0.2))  
  #hidden layer #2
  model.add(Dense(units = (128)))
  model.add(BatchNormalization())
  model.add(Activation('relu'))
  model.add(Dropout(0.2))  
  #hidden layer #3
  model.add(Dense(units = (64)))
  model.add(BatchNormalization())
  model.add(Activation('relu'))
  model.add(Dropout(0.2))  
  #hidden layer #4
  model.add(Dense(units = (32)))
  model.add(BatchNormalization())
  model.add(Activation('relu'))
  model.add(Dropout(0.2))  
  #hidden layer #5
  model.add(Dense(units = (16)))
  model.add(Activation('relu'))

  #output layer
  model.add(Dense(units=2))
  model.add(Activation('softmax'))

  return model

model =dnn_model()
model.summary()
opti = tf.keras.optimizers.Adam(lr=0.001)
checkpoint_path = "model_checkpoint.ckpt"  #best model save
checkpoint = ModelCheckpoint(checkpoint_path, save_weight_only=True,save_best_only=True, monitor="val_loss",verbose=2)
model.compile(optimizer = opti,loss = 'binary_crossentropy',metrics=['accuracy'])

hist = model.fit(X_train, y_train, epochs=200, verbose=True, batch_size=64, validation_data=(X_valid,y_valid), shuffle=False,callbacks=[checkpoint],)

model.save(filepath+'Dnn_1.h5')

df = pd.DataFrame(hist.history)
df.mean()

# visualization
fig, loss_ax = plt.subplots()

acc_ax = loss_ax.twinx()

loss_ax.plot(hist.history['loss'], 'y', label='train loss')
loss_ax.plot(hist.history['val_loss'], 'r', label='test loss')

acc_ax.plot(hist.history['accuracy'], 'b', label='train acc')
acc_ax.plot(hist.history['val_accuracy'], 'g', label='test acc')

loss_ax.set_xlabel('epoch')
loss_ax.set_ylabel('loss')
acc_ax.set_ylabel('accuray')

loss_ax.legend(loc='upper left')
acc_ax.legend(loc='upper right')

plt.show()

from sklearn.metrics import accuracy_score  #정확도
score = model.evaluate(X_valid, y_valid, verbose=0)
print(int(score[1]*100))

"""# DNN TEST"""

import numpy as np
import pandas as pd
from google.colab import drive
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import ModelCheckpoint
drive.mount('/content/gdrive',force_remount=True)
filepath='/content/gdrive/My Drive/'
test = pd.read_csv(filepath+'instagram_test.csv')

model = load_model(filepath+'Dnn_1.h5')
model.summary()

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
scaler.fit(test)
train = pd.DataFrame(data = scaler.transform(test), columns = test.columns, index = test.index)
y_test = test['#fake']
X_test = test.drop('#fake',axis=1)

print('X_test = ', X_test.shape)
print('y_test = ', y_test.shape)
y_test = tf.keras.utils.to_categorical(y_test, num_classes=2)

from sklearn.metrics import accuracy_score
score = model.evaluate(X_test, y_test, verbose=0)

print(int(score[1]*100))

#테스트 파일 중에서 몇 번부터 몇번을 볼까?
for i in range(1,11):
  if y_test[i][0]>y_test[i][1]:
    print("Real Account")
  else:
    print("Fake Account!!")

# TEST파일 중에서 골라서 예측  
dnn_y = model.predict(X_test[1:11])
for i in range(10):
  if dnn_y[i][0]>dnn_y[i][1]:
    print("Real Account")
  else:
    print("Fake Account!!")


# -*- coding: utf-8 -*-
"""
Created on Thu Jun 24 16:01:12 2021
@author: Ricky
"""
#%% Import library
import os
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Conv2D, Dense, Dropout, Flatten, InputLayer, MaxPooling2D, BatchNormalization
import matplotlib.pyplot as plt
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

#%% load data
num_classes = 10
x_train=np.load('./data_train.npy')
y_train=np.load('./label_train.npy')
y_train = keras.utils.to_categorical(y_train, num_classes)

#%% define model
model = Sequential()
model.add(InputLayer(input_shape=(64,64,3)))

model.add(Conv2D(16, kernel_size=(5,5), padding='same', activation="relu"))
# model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
# model.add(Dropout(0.25))

model.add(Conv2D(36, kernel_size=(5,5), padding='same', activation="relu"))
# model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(128, activation='relu'))
# model.add(BatchNormalization())
model.add(Dropout(0.5))

model.add(Dense(num_classes, activation="softmax"))
model.summary()

#%% Train and evaluate
batch_size = 100
epochs = 20
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
train_history = model.fit(x_train, y_train, validation_split=0.2, batch_size=batch_size, epochs=epochs)

#%% Plot training history
plt.plot(train_history.history['loss'])
plt.plot(train_history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='best')
plt.show()

plt.plot(train_history.history['accuracy'])
plt.plot(train_history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='best')
plt.show()

#%% Save model
model.save('C:/Users/Ricky/Desktop/CNN_Model.h5')
print('Model saved!')

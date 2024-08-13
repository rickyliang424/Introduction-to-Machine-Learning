#%% Import library
import numpy as np
import keras
from keras import layers
from keras import callbacks
import os
import matplotlib.image as img
import matplotlib.pyplot as plt
# print(keras.__version__)

#%% load data and preprocessing
num_classes = 10
input_shape = (28,28,1)
(x_train, y_train), (x_test, y_test) = keras.datasets.fashion_mnist.load_data()
x_train = x_train.astype("float32") / 255
x_test = x_test.astype("float32") / 255
x_train = np.expand_dims(x_train, -1)
x_test = np.expand_dims(x_test, -1)

x_train=x_train[0:10000,:,:,:]
y_train=y_train[0:10000]

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

#%% define model
model = keras.Sequential(
    [
        layers.InputLayer(input_shape=input_shape),  # different from ppt beware!!
        layers.Conv2D(128, kernel_size=(3,3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2,2)),
        layers.Conv2D(64, kernel_size=(3,3), activation="relu"),
        layers.BatchNormalization(axis=3),  # different from ppt beware!!
        layers.MaxPooling2D(pool_size=(2,2)),
        layers.Flatten(),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation="softmax"),
    ]
)
model.summary()

#%%training and evaluate
callback_earlystop = callbacks.EarlyStopping(monitor='val_loss', patience=10, min_delta=0.5)

batch_size = 100
epochs = 5
# or "categorical_crossentropy" or "adagrad SGD"
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

history = model.fit(x_train, y_train, validation_split=0.1, batch_size=batch_size, 
                    epochs=epochs, callbacks=[callback_earlystop])

#%% plot
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='best')
plt.show()

test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
print('test_loss=', test_loss, 'test_accuracy=', test_accuracy)

# model.save('MNIST_CNN.h5')

# model=keras.models.load_model('MNIST_CNN.h5')
# print('done')

'''Trains a simple deep NN on the MNIST dataset.

Gets to 98.40% test accuracy after 20 epochs
(there is *a lot* of margin for parameter tuning).
2 seconds per epoch on a K520 GPU.
'''

from __future__ import print_function

import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import RMSprop
import numpy as np
from sklearn import preprocessing

batch_size = 3
num_classes = 3
epochs = 30

x_train = np.array([
    # Up-trend
    [1.0, 2.0, 3.0, 4.0],
    [1.9, 2.0, 3.1, 4.0],
    [1.9, 2.9, 3.9, 5.0],
    [2.1, 3.0, 4.1, 5.0],

    # Down-trend
    [4.0, 3.0, 2.0, 1.0],
    [4.0, 3.0, 2.1, 1.0],
    [5.1, 3.9, 3, 2.0],
    [2.1, 3.0, 4.1, 5.0],

    # no-trend
    [1.0, 1.0, 1.0, 1.0],
    [1.0, 3.0, 1.0, 2.0],
    [1.0, 2.0, 1.0, 2.0],
    [1.0, 3.0, 2.0, 2.5],
    [1.0, 2.0, 0.0, 3.0]
    ])


y_train = np.array([
    [1.0,0.0,0],
    [1.0,0,0],
    [1.0,0,0],
    [1.0,0,0],
    [1.0,0,0],

    [0,1.0,0],
    [0,1.0,0],
    [0,1.0,0],
    [0,1.0,0],
    [0,1.0,0],

    [0,0,1.0],
    [0,0,1.0],
    [0,0,1.0],
    [0,0,1.0],
    [0,0,1.0]
    ])


x_test = np.array([
    # Up-trend
    [1.0 ,1.5, 1.3, 1.8],
    [0.1 ,0.15, 0.13, 0.18],

    # Down-trend
    [1.0, 0.8, 0.9, 0.7],
    [0.1, 0.08, 0.09, 0.07],

    # no-trend
    [1.0, 3.0, 2.0, 2.5],
    [0.1, 0.3, 0.2, 0.25]
    ])

x_train = preprocessing.scale(x_train, axis=1)
x_test = preprocessing.scale(x_test, axis=1)
print(x_train, x_test)

model = Sequential()
model.add(Dense(200, activation='relu', input_shape=(4,)))
model.add(Dense(num_classes, activation='softmax'))

model.summary()

model.compile(loss='categorical_crossentropy',
              optimizer=RMSprop(),
              metrics=['accuracy'])

history = model.fit(x_train, y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1)#, validation_data=(x_test, y_test))

print(model.predict(x_test))
#score = model.evaluate(x_test, y_test, verbose=0)
#print('Test loss:', score[0])
#print('Test accuracy:', score[1])

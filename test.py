import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import RMSprop, Adam
from sklearn import preprocessing

def gendata(validate_split=0.7):

    num = 20
    np.random.seed(111)
    x = np.arange(1, num+1)

    datax = []
    datay = []

    for i in range(1000):
        rand = np.random.random()
        y = pow(1.08, x) + np.sin(3.14*rand*x+2*3.14*rand)*rand/2
        datax.append(y)
        datay.append([1.0, 0, 0, 0, 0])
        rand = np.random.random()
        y = -pow(1.08, x) + np.sin(3.14*rand*x+2*3.14*rand)*rand/2
        datax.append(y)
        datay.append([0, 1.0, 0, 0, 0])

        y = np.log(x) + np.sin(3.14 * rand * x + 2 * 3.14 * rand) * rand / 2
        datax.append(y)
        datay.append([0, 0, 1.0, 0, 0])
        rand = np.random.random()
        y = -np.log(x) + np.sin(3.14 * rand * x + 2 * 3.14 * rand) * rand / 2
        datax.append(y)
        datay.append([0, 0, 0, 1.0, 0])

        y = np.sin(3.14 * rand * x + 2 * 3.14 * rand) * rand / 2
        datax.append(y)
        datay.append([0, 0, 0, 0,1.0])

    assert len(datax) == len(datay)

    datax = preprocessing.scale(datax, axis=1)

    split_index = int(len(datax)*validate_split)
    x = np.split(datax, [split_index])
    y = np.split(datay, [split_index])
    return (x[0], y[0]), (x[1], y[1])


def train():
    batch_size = 64
    num_classes = 2
    epochs = 10

    (x_train, y_train), (x_test, y_test) = gendata()
    input_cols = x_train.shape[1]
    num_classes = y_train.shape[1]

    model = Sequential()
    model.add(Dense(512, activation='relu', input_shape=(input_cols,)))
    model.add(Dropout(0.2))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(num_classes, activation='softmax'))

    model.summary()

    model.compile(loss='categorical_crossentropy',
    #model.compile(loss='mean_squared_error',
                  #optimizer=RMSprop(),
                  optimizer=Adam(),
                  metrics=['accuracy'])

    history = model.fit(x_train, y_train,
                        batch_size=batch_size,
                        epochs=epochs,
                        verbose=1,
                        validation_data=(x_test, y_test)) #, shuffle=True)

    score = model.evaluate(x_test, y_test, verbose=1)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

    # list all data in history
    print(history.history.keys())
    # summarize history for accuracy
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

train()


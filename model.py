import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import RMSprop, Adam
from sklearn import preprocessing
from train_data import get_train_data

def get_model(num=10):
    batch_size = 64
    num_classes = 2
    epochs = 10

    (x_train, y_train), (x_test, y_test) = get_train_data(num, 0.8)
    print(len(x_train), len(x_test))
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

    '''
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
    '''
    return model

if __name__ == '__main__':
    get_model(10)

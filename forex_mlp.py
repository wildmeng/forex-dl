
from __future__ import print_function

import sys
import os
import pandas as pd

import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import RMSprop
from keras.models import model_from_json

import gendata as data

model_file_name = "mlp_model.json"
model_weights_file_name = "mlp_model.h5"

data_file = "eurusd-60min.csv"

period=100

def train_mlp():
    batch_size = 256
    num_classes = 3
    epochs = 5

    (x_train, y_train), (x_test, y_test) = data.gendata(data_file)
    input_cols = x_train.shape[1]
    num_classes = y_train.shape[1]

    model = Sequential()
    model.add(Dense(256, activation='relu', input_shape=(input_cols,)))
    model.add(Dropout(0.2))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(num_classes, activation='softmax'))

    model.summary()

    model.compile(loss='categorical_crossentropy',
                  optimizer=RMSprop(),
                  metrics=['accuracy'])

    history = model.fit(x_train, y_train,
                        batch_size=batch_size,
                        epochs=epochs,
                        verbose=1,
                        validation_data=(x_test, y_test))

    score = model.evaluate(x_test, y_test, verbose=1)

    #data.showdata(sys.argv[1], 1)

    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

    # serialize model to JSON
    model_json = model.to_json()
    with open(model_file_name, "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights(model_weights_file_name)
    print("Saved model to %s"%model_file_name)


def import_model():
    if not os.path.isfile(model_file_name) or not os.access(model_file_name, os.R_OK):
        print("model doesn't exist")
        return None

    json_file = open(model_file_name, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights(model_weights_file_name)
    print("Loaded model from disk")

    # evaluate loaded model on test data
    loaded_model.compile(loss='categorical_crossentropy',
               optimizer=RMSprop(),
               metrics=['accuracy'])
    return loaded_model

if sys.argv[1] == "train":
    train_mlp()
elif sys.argv[1] == "data-x":
    data.gen_x_data("eurusd-60min.csv")
elif sys.argv[1] == "data-y":
    data.gen_y_data("eurusd-60min.csv")
elif sys.argv[1] == "check":
    index = int(sys.argv[2])
    model = import_model()
    (x_train, y_train), (x_test, y_test) = data.gendata(data_file, period_num=100)
    result = model.predict(x_test[index:index+1])
    #result = model.predict(x_test)
    #print(result)
    data.showdata(data_file, index, result[0])


#!/usr/bin/env python

from __future__ import print_function

import sys
import os
import pandas as pd

import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import RMSprop, Adam
from keras.models import model_from_json

import gendata as data


model_file_name = "mlp_model.json"
model_weights_file_name = "mlp_model.h5"

data_file = "eurusd-1m/DAT_ASCII_EURUSD_M1_2016.csv"


def train_mlp():
    batch_size = 256
    epochs = 10

    (x_train, y_train), (x_test, y_test) = data.gendata(data_file)
    input_cols = x_train.shape[1]
    #num_classes = y_train.shape[1]

    model = Sequential()
    model.add(Dense(256, activation='relu', input_shape=(input_cols,)))
    model.add(Dropout(0.2))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(1))
    # model.add(Dense(num_classes, activation='softmax'))

    model.summary()

    model.compile(optimizer='adam', loss='mse')

    '''
    model.compile(loss='categorical_crossentropy',
    #model.compile(loss='mean_squared_error',
                  #optimizer=RMSprop(),
                  optimizer=Adam(),
                  metrics=['accuracy'])
    '''
    history = model.fit(x_train, y_train,
                        batch_size=batch_size,
                        epochs=epochs,
                        verbose=1,
                        validation_data=(x_test, y_test)) #, shuffle=True)

    score = model.evaluate(x_test, y_test, verbose=1)


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
    '''
    loaded_model.compile(loss='categorical_crossentropy',
	       optimizer=Adam(),
               #optimizer=RMSprop(),
               metrics=['accuracy'])
    '''
    loaded_model.compile(optimizer='adam', loss='mse')

    return loaded_model

if sys.argv[1] == "train":
    train_mlp()
elif sys.argv[1] == "datax":
    data.gen_x_data(data_file)
elif sys.argv[1] == "datay":
    data.gen_y_data(data_file)
elif sys.argv[1] == "show":
    if len(sys.argv) != 3:
        print("missing parameters")
        sys.exit(1)

    index = int(sys.argv[2])
    print(index)
    model = import_model()
    (x_train, y_train), (x_test, y_test) = data.gendata(data_file)
    result = model.predict(x_test[index:index+100])
    print(y_test[index:index+100])
    print(result)
    #data.showdata(data_file, 0)


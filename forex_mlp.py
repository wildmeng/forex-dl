
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

def train_mlp():
    batch_size = 128
    num_classes = 3
    epochs = 10

    (x_train, y_train), (x_test, y_test) = data.gendata(sys.argv[1])
    input_cols = x_train.shape[1]

    model = Sequential()
    model.add(Dense(512, activation='relu', input_shape=(input_cols,)))
    model.add(Dropout(0.2))
    model.add(Dense(512, activation='relu'))
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


if os.path.isfile(model_file_name) and os.access(model_file_name, os.R_OK):

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

    (x_train, y_train), (x_test, y_test) = data.gendata(sys.argv[1])
    score = loaded_model.evaluate(x_test, y_test, verbose=0)
    print("%s: %.2f%%" % (loaded_model.metrics_names[1], score[1]*100))

    score = loaded_model.evaluate(x_train, y_train, verbose=0)
    print("%s: %.2f%%" % (loaded_model.metrics_names[1], score[1]*100))
else:
    train_mlp()


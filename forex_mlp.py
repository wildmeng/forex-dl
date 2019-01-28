
from __future__ import print_function

import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import RMSprop
import pandas as pd
import sys

batch_size = 128
num_classes = 3
epochs = 20

fx = pd.read_csv(sys.argv[1].split(".")[0] + "-x.csv")
fy = pd.read_csv(sys.argv[1].split(".")[0] + "-y.csv")

x = fx.values
y = fy.values
total = len(x)
print("total = %d"%total)
print(type(x))
if len(x) != len(y):
    print("x!=y")
    sys.exit()

split_index = int(total*0.9)
print("split at %d" % split_index)
x_train = x[0:split_index]
x_test = x[split_index:]
y_train = y[0:split_index]
y_test = y[split_index:]

model = Sequential()
model.add(Dense(512, activation='relu', input_shape=(800,)))
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
print('Test loss:', score[0])
print('Test accuracy:', score[1])

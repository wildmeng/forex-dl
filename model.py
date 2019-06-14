import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import RMSprop, Adam
from sklearn import preprocessing
# from train_data import get_train_data
from train_data import get_train_data
import plotly as py
import plotly.graph_objs as go
from plotly.offline import plot, iplot

def get_model(num=10):
    batch_size = 64
    num_classes = 2
    epochs = 20

    (x_train, y_train), (x_test, y_test) = get_train_data(num, 0.8)
    print(len(x_train), len(x_test))
    input_cols = x_train.shape[1]
    num_classes = y_train.shape[1]

    model = Sequential()
    model.add(Dense(512, activation='relu', input_shape=(input_cols,)))
    model.add(Dropout(0.2))
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(num_classes, activation='softmax'))

    # model.summary()

    model.compile(loss='categorical_crossentropy',
    #model.compile(loss='mean_squared_error',
                  #optimizer=RMSprop(),
                  optimizer=Adam(),
                  metrics=['accuracy'])

    history = model.fit(x_train, y_train,
                        batch_size=batch_size,
                        epochs=epochs,
                        verbose=0,
                        validation_data=(x_test, y_test), shuffle=True)

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
    '''
    return model


def showdata(p = 20):
    model = get_model(p)
    csv_file = "data/XBTUSD_5m_70000_train.csv"
    df = pd.read_csv(csv_file)



    for start in range(1000, 1200):
        end = start + p
        print(start, end)
        trace = go.Ohlc(#x=df['DTYYYYMMDD'],
                        open=df.open[start: end],
                        high=df.high[start: end],
                        low=df.low[start: end],
                        close=df.close[start: end])


        #x = [i for i range(100)]

        trace2 = go.Scatter(y=df.close[start: end])
        data = [trace, trace2]

        x=df.close[start: end]
        x = np.array(x)
        x = np.reshape(x, (1,p))
        x = preprocessing.scale(x, axis=1)
        trend = model.predict(x)

        #print(trend)
        #names = ['up', 'down', 'flat', 'down2flat', 'up2flat', 'flat2up', 'flat2down', 'up2down', 'down2up']
        names = ['up', 'down', 'flat']
        index = np.argmax(trend[0])
        #print(names[index], p[0][index])
        py.offline.plot(data, filename='data/%d-test-%s-%.2f.html'%(start, names[index], trend[0][index]), auto_open=False)


if __name__ == '__main__':
    #
    showdata(20)

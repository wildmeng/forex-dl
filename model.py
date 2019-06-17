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
    batch_size = 32
    num_classes = 2
    epochs = 40

    x_train, y_train = get_train_data(num, 0.8)
    input_cols = x_train.shape[1]
    num_classes = y_train.shape[1]

    model = Sequential()
    model.add(Dense(64, activation='relu', input_shape=(input_cols,)))
    #model.add(Dropout(0.2))
    model.add(Dense(32, activation='relu'))
    #model.add(Dropout(0.2))
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
                        verbose=1, shuffle=True)

    #score = model.evaluate(x_test, y_test, verbose=1)

    plt.plot(history.history['loss'])
    #plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    #plt.legend(['train', 'test'], loc='upper left')
    #plt.show()

    return model


def showdata(p = 20):
    model = get_model(p)
    csv_file = "data/XBTUSD_5m_70000_train.csv"
    df = pd.read_csv(csv_file)



    for start in range(1500, 1700):
        end = start + p
        print(start, end)
        '''
        trace = go.Ohlc(#x=df['DTYYYYMMDD'],
                        open=df.open[start: end],
                        high=df.high[start: end],
                        low=df.low[start: end],
                        close=df.close[start: end])

        #x = [i for i range(100)]
        '''

        x=df.close[start: end]
        x = np.array(x)
        x = np.reshape(x, (1,p))
        x = preprocessing.scale(x, axis=1)
        trend = model.predict(x)

        x = np.reshape(x, (p,1))
        print(x)
        trace2 = go.Scatter(y=x)
        data = [trace2]  # [trace, trace2]

        #print(trend)
        #names = ['up', 'down', 'flat']
        names = ['up', 'down', 'flat']
        index = np.argmax(trend[0])
        #print(names[index], p[0][index])
        py.offline.plot(data, filename='data/%d-test-%s-%.2f.html'%(start, names[index], trend[0][index]), auto_open=False)


if __name__ == '__main__':
    model = get_model(20)
    '''
    x = np.array([1,2,3,4,5,6,7,8,9,10,11,12,13,14,13,12,11,10,9,8])
    x = np.reshape(x, (1,20))
    x = preprocessing.scale(x, axis=1)
    trend = model.predict(x)
    print(trend)
    '''
    showdata(20)

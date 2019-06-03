import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import preprocessing
from sklearn.utils import shuffle


def notrend2down(x, p):
    y = np.array([0.0]*len(x))
    total = len(x)
    for i in range((total*8)//10, total):
        y[i] = y[i-1] + (1/p)
    return y

def gendata(num, f, p, datax, notrend=False):
    np.random.seed(0)
    x = np.arange(0, num)
    feature_num = 10
    y = f(x, p) # pow(1.1, x)
    hight = abs(y[num - 1] - y[0])
    if hight > 0 and not notrend:
        amp = np.linspace(0.2, 0.5, feature_num)*hight
    else:
        amp = [1]

    cycle = np.linspace(3, 10, feature_num)
    shift = np.linspace(0, 2*np.pi, feature_num, endpoint=False)

    for s in shift:
        for a in amp:
            for c in cycle:
                T = (2*np.pi/num)*c
                y1 = a*np.sin(T*x + s) + a
                datax.append(y + y1)
                print(y)
                plt.plot(x,y+y1)
                plt.show()
    # plt.show()
    return (len(shift))*len(amp)*len(cycle)

def get_train_data(period = 20, split_at = 0.8):
    datax = []
    datay = []

    gendata(period, notrend2down, 3, datax, True)

if __name__ == '__main__':
    get_train_data()
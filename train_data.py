import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import preprocessing
from sklearn.utils import shuffle

def uptrend1(x, p):
    return x/p

def up2downtrend1(x, p):
    y= x/p
    total = len(x)
    for i in range((total*8)//10, total):
        y[i] = y[i-1] - (1/p)
    return y

def down2uptrend1(x, p):
    y= -x/p
    total = len(x)
    for i in range(total*4//5, total):
        y[i] = y[i-1] + (1/p)
    return y

def downtrend1(x, p):
    return -x/p

def uptrend2(x, p):
    return pow(1.1, x)

def downtrend2(x, p):
    return -pow(1.1, x)

def up2downtrend2(x, p):
    y= pow(1.1, x)
    total = len(x)
    for i in range(total*4//5, total):
        y[i] = y[i-1]/p

    return y

def notrend2down(x, p):
    y = np.array([0.0]*len(x))
    total = len(x)
    for i in range((total*8)//10, total):
        y[i] = y[i-1] - (1/p)
    return y

def notrend2up(x, p):
    y = np.array([0.0]*len(x))
    total = len(x)
    for i in range((total*8)//10, total):
        y[i] = y[i-1] + (1/p)
    return y

def down2uptrend2(x, p):
    y= -x/p
    total = len(x)
    for i in range(total*4//5, total):
        y[i] = y[i-1]*p
    return y

def notrend(x, p):
    return np.array([0.0]*len(x))

# Generate the training data for various patterns,
# which is represented by the input f (function)
def gendata(num, f, p, datax, notrend=False):
    np.random.seed(0)
    x = np.arange(0, num)
    feature_num = 10
    y = f(x, p) # pow(1.1, x)
    hight = abs(y[num - 1] - y[0])
    if not notrend:
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
                #plt.plot(x,y+y1)
                #plt.show()
    # plt.show()
    return (len(shift))*len(amp)*len(cycle)

def get_train_data(period = 20, split_at = 0.8):
    datax = []
    datay = []

    cnt = gendata(period, uptrend1, 3, datax)
    datay += ([[1,0,0,0,0,0,0]] * cnt)
    cnt = gendata(period, downtrend1, 3, datax)
    datay += ([[0,1,0,0,0,0,0]] * cnt)

    cnt = gendata(period, up2downtrend1, 3, datax)
    datay += ([[0,0,1,0,0,0,0]] * cnt)
    cnt = gendata(period, down2uptrend1, 3, datax)
    datay += ([[0,0,0,1,0,0,0]] * cnt)

    cnt = gendata(period, uptrend2, 1.1, datax)
    datay += ([[1,0,0,0,0,0,0]] * cnt)
    cnt = gendata(period, downtrend2, 1.1, datax)
    datay += ([[0,1,0,0,0,0,0]] * cnt)

    cnt = gendata(period, up2downtrend2, 1.1, datax)
    datay += ([[0,0,1,0,0,0,0]] * cnt)

    cnt = gendata(period, down2uptrend1, 1.1, datax)
    datay += ([[0,0,0,1,0,0,0]] * cnt)

    cnt = gendata(period, notrend, 0, datax, True)
    datay += ([[0,0,0,0,1,0,0]] * cnt)

    cnt = gendata(period, notrend2up, 3, datax, True)
    datay += ([[0,0,0,0,0,1,0]] * cnt)
    cnt = gendata(period, notrend2down, 3, datax, True)
    datay += ([[0,0,0,0,0,0,1]] * cnt)

    assert(len(datax) == len(datay))
    x = np.array(datax)
    y = np.array(datay)
    x = preprocessing.scale(x, axis=1)
    x, y = shuffle(x, y, random_state=0)

    split_index = int(len(datax) * split_at)
    sx = np.split(x, [split_index])
    sy = np.split(y, [split_index])
    return (sx[0], sy[0]), (sx[1], sy[1])

if __name__ == '__main__':
    get_train_data()

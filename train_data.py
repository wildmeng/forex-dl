import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.utils import shuffle

def up(num, shift):
    p = 0.5
    x = np.linspace(0.0, 4*np.pi, num=num)
    y = x*p + np.sin(x+shift)

    return y

def down(num, shift):
    p = -0.5
    x = np.linspace(0.0, 4*np.pi, num=num)
    y = x*p + np.sin(x+shift)

    return y


def flat1(num, shift):
    x = np.linspace(0.0, 4*np.pi, num=num)
    y = np.sin(x+shift)

    return y

def flat2(num, shift):
    x = np.linspace(0.0, 4*np.pi, num=num)
    y = (np.power(1.05, -x))*np.sin(x+shift)

    return y

def flat3(num, shift):
    x = np.linspace(0.0, 4*np.pi, num=num)
    y = (np.power(1.05, x))*np.sin(x+shift)

    return y

def get_train_data(period = 20, split_at = 0.8):
    trends = [
        up,
        down,
        flat1,
        flat2,
        flat3
    ]

    x = []
    y = []
    vects = np.eye(len(trends))
    for i, t in enumerate(trends):
        shifts = np.linspace(0.0, np.pi, 10)
        for shift in shifts:
            result = t(period, shift)
            x.append(result)
            y.append(vects[i].tolist())

    assert(len(x) == len(y))
    print("total training set:",len(x))

    x = np.array(x)
    y = np.array(y)
    x = preprocessing.scale(x, axis=1)
    x, y = shuffle(x, y) #, random_state=0)

    return x, y

    '''
    split_index = int(len(x) * split_at)
    sx = np.split(x, [split_index])
    sy = np.split(y, [split_index])
    return (sx[0], sy[0]), (sx[1], sy[1])
    '''

if __name__ == '__main__':
    get_train_data()
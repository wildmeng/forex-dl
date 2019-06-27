import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.utils import shuffle
import math
import itertools

def up1(num, shift=0, cycles = 2, mag = 0.3):
    x = np.linspace(0.0, 2*cycles*np.pi, num=num)
    y = np.linspace(0, 1, num)
    y += mag*np.sin(x+shift)

    return y

def up2(num, shift=0, cycles = 2, mag = 0.3):
    x = np.linspace(0.0, 2*cycles*np.pi, num=num)
    y = np.geomspace(1, 2, num)
    y += mag*np.sin(x+shift)

    return y

def down1(num, shift=0, cycles = 2, mag = 0.3):
    x = np.linspace(0.0, 2*cycles*np.pi, num=num)
    y = np.linspace(1, 0, num)
    y += mag*np.sin(x+shift)

    return y

def down2(num, shift=0, cycles = 2, mag = 0.3):
    x = np.linspace(0.0, 2*cycles*np.pi, num=num)
    y = np.geomspace(2, 1, num)
    y += mag*np.sin(x+shift)

    return y

def flat1(num, shift, cycles = 2):
    x = np.linspace(0.0, cycles*2*np.pi, num=num)
    y = np.sin(x+shift)

    return y

def flat2(num, shift=0, cycles = 2):
    y = np.linspace(1, 0, num)
    x = np.linspace(0.0, cycles*2*np.pi, num=num)
    y = np.sin(x+shift)*y

    return y

def flat3(num, shift=0, cycles = 2):
    y = np.linspace(0, 1, num)
    x = np.linspace(0.0, cycles*2*np.pi, num=num)
    y = np.sin(x+shift)*y

    return y

def flat4(num, shift, cycles = 2):
    y1 = np.linspace(0, 1, num/2)
    y2 = np.linspace(1, 0, num-num/2)
    x = np.linspace(0.0, cycles*2*np.pi, num=num)
    y = np.concatenate((y1,y2))
    y = np.sin(x+shift)*y

    return y

def flat5(num, shift=0, cycles = 4):
    y1 = np.linspace(1, 0, num)
    x = np.linspace(0.0, cycles*2*np.pi, num=num)

    y = np.sin(x+shift)
    y += 1
    y = y*y1

    return y

def flat6(num, shift=0, cycles = 4):
    y1 = np.linspace(0, 1, num)
    x = np.linspace(0.0, cycles*2*np.pi, num=num)

    y = np.sin(x+shift)
    y += 1
    y = y*y1

    return y

def add_trends(trends, period):
    fig_num = 0

    x = []
    y = []
    vects = np.eye(len(trends))
    for i, t in enumerate(trends):
        for ts in t:
            for c in range(2, period//4):
                shifts = np.linspace(0.0, np.pi, 10)
                for shift in shifts:
                    result = ts(period, shift, c)
                    result = preprocessing.minmax_scale(result)
                    x.append(result)
                    y.append(vects[i].tolist())
                    # save plot
                    #x1 = np.linspace(0.0, c*2*np.pi, period)
                    #plt.plot(x1, result,'--bo')
                    #plt.savefig('./train-data/%d-%d.png'%(i, fig_num))
                    #plt.clf()
                    fig_num += 1

    return x, y


def merge_trend(f1, f2, num, cycles, split_cycles):
    n = int(num*split_cycles//cycles)
    y1 = f1(n, 0, split_cycles)
    y2 = f2(num-n, np.pi, cycles - split_cycles)

    y2 += (y1[n-1] - y2[0])

    return list(y1) + list(y2)


def blend_trends(trends, period):
    all_trends = list(itertools.permutations(trends, 2))
    for t in all_trends:
        paires = list(itertools.product(t[0], t[1]))
        for pair in paires:
            print(pair)
            x1 = np.linspace(0.0, 4*2*np.pi, period)
            y = merge_trend(pair[0], pair[1], period, 6, 4)
            plt.plot(x1, y,'--bo')
            plt.show()


def get_train_data(period = 20, split_at = 0.8):
    trends = [
        [up1, up2],
        [down1, down2],
        [flat1,flat2,flat3,flat4, flat5, flat6]
    ]


    blend_trends(trends, period)
    '''
    x, y = add_trends(trends, period)

    assert(len(x) == len(y))
    print("total training set:",len(x))

    x = np.array(x)
    y = np.array(y)

    #x = preprocessing.minmax_scale(x, axis=1)
    x, y = shuffle(x, y) #, random_state=0)

    return x, y
    '''


if __name__ == '__main__':
    get_train_data()
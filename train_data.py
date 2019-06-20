import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.utils import shuffle
import math
import itertools

def up1(num, shift, cycles = 2):
    p = 0.5
    x = np.linspace(0.0, 2*cycles*np.pi, num=num)
    y = x*p + np.sin(x+shift)

    return y

def up2(num, shift, cycles = 2):
    p = 1.06
    x = np.linspace(0.0, 2*cycles*np.pi, num=num)
    y = np.power(p,x) + np.sin(x+shift)
    return y

def down1(num, shift, cycles = 2):
    p = -0.5
    x = np.linspace(0.0, 2*cycles*np.pi, num=num)
    y = x*p + np.sin(x+shift)

    return y

def down2(num, shift, cycles = 2):
    p = 1.06
    x = np.linspace(0.0, cycles*2*np.pi, num=num)
    y = -np.power(p,x) + np.sin(x+shift)
    return y

def flat1(num, shift, cycles = 2):
    x = np.linspace(0.0, cycles*2*np.pi, num=num)
    y = np.sin(x+shift)

    return y

def flat2(num, shift, cycles = 2):
    x = np.linspace(0.0, cycles*2*np.pi, num=num)
    y = (np.power(1.05, -x))*np.sin(x+shift)

    return y

def flat3(num, shift, cycles = 2):
    x = np.linspace(0.0, cycles*2*np.pi, num=num)
    y = (np.power(1.05, x))*np.sin(x+shift)

    return y

def flat4(num, shift, cycles = 2):
    xlen = 7*np.pi
    def f1(x):
        return (np.power(1.03, x)) * np.sin(x)

    a = math.pow(1.03, xlen/2)

    def f2(x):
        return (np.power(1.03, -(x - xlen/2))) * np.sin(x) * a

    x1 = np.linspace(0.0, cycles*2*np.pi, num=num)
    y = np.piecewise(x1, [x1 < xlen / 2, x1 >= xlen/2], [f1, f2])
    return y

def add_trends(trends, period):
    fig_num = 0

    x = []
    y = []
    vects = np.eye(len(trends))
    for i, t in enumerate(trends):
        for ts in t:
            for c in range(2, 5):
                shifts = np.linspace(0.0, np.pi, 10)
                for shift in shifts:
                    result = ts(period, shift, c)
                    result = preprocessing.minmax_scale(result)
                    x.append(result)
                    y.append(vects[i].tolist())
                    # save plot
                    x1 = np.linspace(0.0, c*2*np.pi, period)
                    plt.plot(x1, result,'--bo')
                    plt.savefig('./train-data/%d-%d.png'%(i, fig_num))
                    plt.clf()
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



def get_train_data(period = 40, split_at = 0.8):
    trends = [
        [up1, up2],
        [down1, down2],
        [flat1,flat2,flat3,flat4]
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
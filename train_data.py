import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.utils import shuffle

def up(num):
    p = 1.1
    x = np.arange(0, num)
    y = pow(p, x) - 1.0

    return y

def down(num):
    p = 1.1
    x = np.arange(0, num)
    return -pow(p, x) + 1.0

def up2(num):
    p = 1/3
    x = np.arange(0, num)
    y = x*p

    return y

def down2(num):
    p = 1/3
    x = np.arange(0, num)
    y = -x*p

    return y

def up3(num):
    x = np.arange(0, num)
    y = np.power(x, 0.65)
    return y

def down3(num):
    x = np.arange(0, num)
    y = -np.power(x, 0.65)
    return y

def notrend(num):
    return np.array([0.0]*num)

def merge_trend(num, f, ratio):
    y1 = f[0](num)
    y2 = f[1](num)

    start = int(num*ratio)
    y2 += (y1[start-1])
    for i in range(start, num):
         y1[i] =y2[i-start]

    return y1

def add_wave_on_trend(trend_base, trend_type, amps=10, cycles=10, shifts=10):

    np.random.seed(0)
    num = len(trend_base)

    output_x = []
    output_y = []

    hight = abs(trend_base[num - 1] - trend_base[0])
    if amps > 0:
        amp = np.linspace(0.2, 0.5, amps) * hight
    else:
        amp = [1.0]

    cycle = np.linspace(3, 10, cycles)
    shift = np.linspace(0, 2*np.pi, shifts, endpoint=False)
    x = np.arange(0, num)
    for s in shift:
        for a in amp:
            for c in cycle:
                T = (2*np.pi/num)*c
                wave = a*np.sin(T*x + s) + a
                output_x.append(trend_base+wave)
                output_y.append(trend_type)
                #plt.plot(x,trend_base+wave,'--bo')
                #plt.show()

    return output_x, output_y

def _gen_data(period, configs):
    datax = []
    datay = []
    trend_cnt = len(configs)
    for i, config in enumerate(configs):
        trend_vec = [0.0] * trend_cnt
        trend_vec[i] = 1.0

        if len(config) < 2:
            for f in config[0]:
                base = f(period)
                amps = 10
                if (f == notrend):
                    amps = 0
                x, y = add_wave_on_trend(base, trend_vec, amps = amps)
                datax += x
                datay += y
        else:
            for f1 in config[0]:
                for f2 in config[1]:
                    amps = 10
                    if f1 == notrend or f2 == notrend:
                        amps = 0
                    base = merge_trend(period, [f1,f2], config[2])
                    x, y = add_wave_on_trend(base, trend_vec, amps=amps)
                    datax += x
                    datay += y

    return datax, datay

def get_train_data(period = 20, split_at = 0.8):
    trends = [
        [[up3, up2, up]],
        [[down3, down2, down]],
        [[notrend]],

        [[down2, down, down3], [notrend], 0.8],
        [[up, up2, up3], [notrend], 0.8],
        [[notrend], [up, up2, up3], 0.7],
        [[notrend], [down, down2, down3], 0.8],

        [[up, up2, up3], [down, down2, down3], 0.8],
        [[down, down2, down3], [up, up2, up3], 0.8],
    ]

    x, y = _gen_data(period, trends)

    assert(len(x) == len(y))
    print("total training set:",len(x))

    x = np.array(x)
    y = np.array(y)
    x = preprocessing.scale(x, axis=1)
    x, y = shuffle(x, y) #, random_state=0)

    split_index = int(len(x) * split_at)
    sx = np.split(x, [split_index])
    sy = np.split(y, [split_index])
    return (sx[0], sy[0]), (sx[1], sy[1])

if __name__ == '__main__':
    get_train_data()
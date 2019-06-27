import numpy as np
import matplotlib.pyplot as plt
import math
from sklearn import preprocessing

def add_line(a, p1, p2, num):
    if num == 0:
        #a.append(p2)
        return

    change = (p2-p1)/num

    for i in range(num):
        last = a[-1]
        a.append(change+last)

def f(num, upper_bound, lower_bound):

    a = [upper_bound[0]]

    total_mag = 0.0
    lowers_num = len(lower_bound)
    uppers_num = len(upper_bound)

    for i in range(uppers_num):
        if i >= lowers_num:
            break
        total_mag += math.fabs(upper_bound[i] - lower_bound[i])

        if i+1 < uppers_num:
            total_mag += math.fabs(upper_bound[i+1] - lower_bound[i])

    for i in range(uppers_num):
        if i >= lowers_num:
            break

        num1 = round((num-1)*(math.fabs(upper_bound[i] - lower_bound[i]))/total_mag)
        add_line(a, upper_bound[i], lower_bound[i], num1)

        if i+1 < uppers_num:
            num2 = round((num-1)*(math.fabs(upper_bound[i+1] - lower_bound[i]))/total_mag)
            add_line(a, lower_bound[i], upper_bound[i+1], num2)

    return a

def up():

    for n in range(3, 7):
        for mag in np.linspace(0.5, 0.5, 5):
            bound1 = np.linspace(0.0, 1.0, n) #[1.0, 1.0, 1.0, 1.0]
            bound2 = bound1 + mag


            a = f(20, bound1, bound2)
            print(len(a))
            print(a)
            a = preprocessing.minmax_scale(a)
            print(a)
            plt.plot(a,'--bo')

            plt.show()

            a = f(20, bound2, bound1)
            a = preprocessing.minmax_scale(a)
            print(len(a))
            plt.plot(a,'--bo')
            plt.show()


def flat2(num, shift=-np.pi/2, cycles = 40):
    y1 = np.linspace(0, 1, num)
    x = np.linspace(0.0, cycles*2*np.pi, num=num)

    y = np.sin(x+shift)
    y += 1
    y = y*y1

    return y

#rotate_list(a)
plt.plot(flat2(20),'--bo')
plt.show()
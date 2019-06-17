import matplotlib.pyplot as plt
import numpy as np
import math

len = 16*np.pi


def f1(x):
    return (np.power(1.03, x))*np.sin(x)


a = math.pow(1.03, len/2)*math.sin(len/2)

def f2(x):
    return (np.power(1.03, -x))*np.sin(x) + a

x = np.linspace(0, len, num=40)

y = np.piecewise(x, [x < len/2, x >= len/2], [f1, f2])


plt.plot(x, y,'--bo')
plt.show()
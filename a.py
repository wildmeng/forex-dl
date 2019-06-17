import matplotlib.pyplot as plt
import numpy as np
import math

len = 7*np.pi

p=0.5
x = np.linspace(0.0, 4*np.pi, num=20)
y = x*p + np.sin(x+np.pi)
plt.plot(x, y,'--bo')
plt.show()
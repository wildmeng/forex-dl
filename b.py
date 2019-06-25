import numpy as np
import matplotlib.pyplot as plt
import math

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
        num1 = round(num*(math.fabs(upper_bound[i] - lower_bound[i]))/total_mag)
        add_line(a, upper_bound[i], lower_bound[i], num1)

        if i+1 < uppers_num:
            num2 = round(num*(math.fabs(upper_bound[i+1] - lower_bound[i]))/total_mag)
            add_line(a, lower_bound[i], upper_bound[i+1], num2)

    return a

upper_bound = np.geomspace(1, 3, 7) #[1.0, 1.0, 1.0, 1.0]
lower_bound = np.geomspace(2, 4, 7)#[0.0, 0.0, 0.0, 0.0]

a = f(50, upper_bound, lower_bound)
plt.plot(a,'--bo')
print(len(a))
print(a)
plt.show()
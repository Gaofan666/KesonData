import numpy as np
import matplotlib.pyplot as plt

x = np.arange(-3, 3, 0.1)


def func1(x):
    y = (2 ** (2 / 3) - x ** (2 / 3)) ** (3 / 2)
    return y


y = func1(x)

plt.hlines(0, -3, 3)
plt.vlines(0, -3, 3)
plt.xlim(-3, 3)
plt.ylim(-3, 3)
plt.plot(x, func1(x), color="red", linewidth=1)
plt.show()

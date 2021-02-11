import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

x = np.linspace(0, 10, 1000)
y = tf.sqrt(x*2-x**2)
plt.figure(figsize=(6, 4))
plt.hlines([0.0,0.2],0,2)
plt.plot(x, y, color="red", linewidth=1)
plt.show()

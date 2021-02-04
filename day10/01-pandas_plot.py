import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

s = pd.Series(np.random.normal(10,1,10))
print(s)
s.plot(kind='line')
# s.plot(kind='barh') # 水平柱状图 bar是垂直柱状图
# plt.scatter(x=s.index,y=s,color='red',label='mypoint')
s.plot.bar(rot=30) # 水平轴坐标倾斜30°
plt.legend()
plt.show()
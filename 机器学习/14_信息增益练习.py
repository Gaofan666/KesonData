import math
import matplotlib.pyplot as mp
import numpy as np

print('--------------练习(1)--------------')
print('按形状分，长条状占 4/5  椭圆形占1/5')

s1 = -0.5 * math.log2(0.5) * 2
s2 = 0
print('长条状分支的信息熵为:%f 椭圆状分枝信息熵为:%f' % (s1, s2))

up = 1.522 - 0.8 * s1 - 0
print('信息增益为：%f' % up)

# ----------------------练习（2）--------------------
x = np.arange(0, 1.1, 0.1)
y = []

# 计算信息熵并填入列表
for a in x:
    b = 1 - a
    if a == 0 or a == 1:  # 当样本中只有一类时，信息熵为0
        result = 0
    else:
        result = -(a * math.log2(a) + b * math.log2(b))
    y.append(result)

mp.figure('Entropy')
mp.title('Doubel Sample Entropies', fontsize=16)
mp.ylabel('Entropy')
mp.xlabel('A Percent')
mp.xticks(x)
mp.yticks(y)
mp.grid(linestyle='-')
mp.plot(x, y, c='orangered', label='Entropy')


mp.legend()
mp.show()

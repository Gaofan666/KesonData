import paddle
import paddle.fluid as fluid
import numpy as np
import matplotlib.pylab as plt

# 定义样本
train_data = np.array([[0.5], [0.6], [0.8], [1.1], [1.4]]).astype('float32')
y_true = np.array([[5.0], [5.5], [6.0], [6.8], [6.8]]).astype('float32')

# 定义变量
x = fluid.layers.data(name='x', shape=[1], dtype='float32')
y = fluid.layers.data(name='y', shape=[1], dtype='float32')

# 搭建全连接模型，构建损失函数，优化器
y_predict = fluid.layers.fc(input=x,  # 输入数据
                            size=1,  # 输出值的个数   回归问题size=1
                            act=None)  # 激活函数，回归这里不采用激活函数

# 损失函数
cost = fluid.layers.square_error_cost(input=y_predict,
                                      label=y)

avg_cost = fluid.layers.mean(cost)  # 均方差

optimizer = fluid.optimizer.SGD(learning_rate=0.1)  # 随机梯度优化
optimizer.minimize(avg_cost) # 均方差最小化

# 执行器
place = fluid.CPUPlace()
exe = fluid.Executor(place)
exe.run(fluid.default_startup_program())

# 开始迭代训练
costs = []
iters = []
params = {'x': train_data, 'y': y_true}

for i in range(200):
    outs = exe.run(feed=params,
                   fetch_list=[y_predict.name, avg_cost.name])
    iters.append(i)  # 记录迭代次数
    costs.append(outs[1][0])  # 记录损失值
    print('i:', i, 'cost:', outs[1][0])

# 损失函数可视化
plt.figure('Training')
plt.title('Train Cost', fontsize=24)
plt.xlabel('Iter', fontsize=14)
plt.ylabel('Cost', fontsize=14)
plt.plot(iters, costs, color='red', label='Train Cost')
plt.grid()
plt.savefig('train.png')  # 保存图片

# 线性模型可视化
tmp = np.random.rand(10, 1)  # 生成10行1列的均匀的随机数组
tmp = tmp * 2  # 范围放大
tmp.sort(axis=0)  # 排序
x_test = np.array(tmp).astype('float32')
params = {'x': x_test, 'y': x_test}  # y不参加计算，仅仅是为了避免语法错误
y_out = exe.run(feed=params,
                fetch_list=[y_predict.name])  # 返回预测结果
y_test = y_out[0]
plt.figure('Infer')
plt.title('Linear Regressino', fontsize=24)
plt.plot(x_test, y_test, color='red', label='infer')
plt.scatter(train_data, y_true)

plt.legend()
plt.grid()
plt.savefig('infer.png')
plt.show()

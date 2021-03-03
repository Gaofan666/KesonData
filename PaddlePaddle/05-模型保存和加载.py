# 多元回归
# 数据集：包含506笔房价数据，每笔数据13个特征1个标签

import paddle
import paddle.fluid as fluid
import matplotlib.pylab as plt
import numpy as np
import os

# 1.数据准备
BUF_SIZE = 500
BATCH_SIZE = 20
# 随机读取器
random_reader = paddle.reader.shuffle(paddle.dataset.uci_housing.train(),
                                      buf_size=BUF_SIZE)
#
train_reader = paddle.batch(random_reader, batch_size=BATCH_SIZE)
train_data = paddle.dataset.uci_housing.train()
for sample in train_data():
    print(sample)

# 2.模型的搭建
x = fluid.layers.data(name='x', shape=[13], dtype='float32')
y = fluid.layers.data(name='y', shape=[1], dtype='float32')
# 定义全连接模型
y_predict = fluid.layers.fc(input=x,  # 输入
                            size=1,  # 输出值的葛书
                            act=None)  # 激活函数
# 损失函数
cost = fluid.layers.square_error_cost(input=y_predict,  # 预测值
                                      label=y)  # 真实值/标签值
avg_cost = fluid.layers.mean(cost)  # 均方差
# 优化器
optimizer = fluid.optimizer.SGD(learning_rate=0.001)
optimizer.minimize(avg_cost)  # 指定优化的目标函数

# 3.模型训练，保存
# 定义执行器
place = fluid.CPUPlace()
exe = fluid.Executor(place)
exe.run(fluid.default_startup_program())
# feeder:参数喂入器，能对参数格式进行转换，转为模型所需要的张量格式
feeder = fluid.DataFeeder(place=place, feed_list=[x, y])
iter = 0
iters = []
train_costs = []
EPOCH_NUM = 120
model_save_dir = 'model/uci_housing'

for pass_id in range(EPOCH_NUM):
    train_cost = 0
    i = 0
    for data in train_reader():
        i += 1
        train_cost = exe.run(program=fluid.default_main_program(),
                             feed=feeder.feed(data),
                             fetch_list=[avg_cost])
        if i % 20 == 0:
            print('pass_id: %d , cost:%f' % (pass_id, train_cost[0][0]))
        iter = iter + BATCH_SIZE
        iters.append(iter)
        train_costs.append(train_cost[0][0])  # 记录损失值

# 保存模型
if not os.path.exists(model_save_dir):
    os.makedirs(model_save_dir)
fluid.io.save_inference_model(model_save_dir,
                              ['x'],  # 预测时喂入的参数
                              [y_predict],  # 模型预测结果从哪里获取
                              exe)  # 模型
# 训练进程可视化
plt.figure('Trainning Cost')
plt.title('Training Cost', fontsize=24)
plt.xlabel('iter', fontsize=14)
plt.ylabel('cost', fontsize=14)
plt.plot(iters, train_costs, color='red', label='Training Cost')
plt.grid()
plt.savefig('train.png')

# 4.模型加载，预测
infer_exe = fluid.Executor(place)
infer_result = []  # 预测值的列表
ground_truth = []  # 真实值的列表

# 加载模型
infer_program, feed_target_names, fetch_targets = \
    fluid.io.load_inference_model(model_save_dir,  # 模型保存的路径
                                  infer_exe)  # 要加载到哪个执行器上
# 测试集读取reader
infer_reader = paddle.batch(paddle.dataset.uci_housing.test(),  # 读取测试集
                            batch_size=200)
test_data = next(infer_reader())  # 获取一批数据
test_x = np.array([data[0] for data in test_data]).astype('float32')
test_y = np.array([data[1] for data in test_data]).astype('float32')

# 构建参数字典
x_name = feed_target_names[0]  # 获取参数名称
results = infer_exe.run(infer_program,
                        feed={x_name: test_x},
                        fetch_list=fetch_targets)  # 获取预测结果
# 预测值列表
for i, v in enumerate(results[0]):
    infer_result.append(v)

# 真实值列表
for i ,v in enumerate(test_y):
    ground_truth.append(v)

# 绘制散点图
plt.figure('infer')
plt.title('infer', fontsize=14)
plt.xlabel('ground truth', fontsize=14)
plt.ylabel('infer result', fontsize=14)
x = np.arange(1, 30)
y = x
plt.scatter(ground_truth, infer_result, color='green', label='infer')
plt.grid()
plt.legend()
plt.savefig('predict.png')
plt.show()

import os

import tensorflow as tf

# 1.创建样本数据
x = tf.random_normal([100, 1], mean=1.75, stddev=0.5, name='x_data')  # 均值 方差
y_true = tf.matmul(x, [[2.0]]) + 5.0  # 计算y = 2x+5

# 2.建立线性模型
# 初始化权重（随机数）和偏置（固定设置为0）， 计算wx+b得到预测值
weight = tf.Variable(tf.random_normal([1, 1]), name='w', trainable=True)  # 训练过程中值是否允许变化  这个值是随机的
bias = tf.Variable(0.0, name='b', trainable=True)  # 偏置
y_predict = tf.matmul(x, weight) + bias

# 3.创建损失函数
loss = tf.reduce_mean(tf.square(y_true - y_predict))  # 均方差

# 4.使用梯度下降进行训练  0.1是学习率
train_op = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

# 收集损失函数的值
tf.summary.scalar('losses', loss)
merged = tf.summary.merge_all()  # 合并摘要操作

init_op = tf.global_variables_initializer()

# 保存模型
saver = tf.train.Saver() # 实例化一个saver

with tf.Session() as sess:
    sess.run(init_op)  # 执行初始化op
    # 打印初始权重和偏置
    print('weight:', weight.eval(), 'bias:', bias.eval())

    # 指定事件并记录图的信息
    fw = tf.summary.FileWriter('./summary/', graph=sess.graph)

    # 训练之前，检查是否已经有模型保存，如果有则加载，(每次运行都是在原来训练结果的基础上再训练)
    if os.path.exists('./model/linear_model/checkpoint'):
        saver.restore(sess,'./model/linear_model/')

    # 循环训练
    for i in range(200):  # 500次差不多是 2x+5
        sess.run(train_op)
        summary = sess.run(merged)  # 执行摘要合并操作
        fw.add_summary(summary, i)  # 写入事件文件
        print(i, ':', 'weight:', weight.eval(), 'bias:', bias.eval())

    # 训练完成，保存模型
    saver.save(sess,'./model/linear_model/')
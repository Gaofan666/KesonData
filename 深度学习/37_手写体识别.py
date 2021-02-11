import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import pylab

# 定义样本读取对象
mnist = input_data.read_data_sets('MNIST_data/',  # 数据集所在目录
                                  one_hot=True)  # 标签是否采用独热编码
# 定义占位符，用于表图像数据，标签
x = tf.placeholder(tf.float32, [None, 784])  # 图像数据
y = tf.placeholder(tf.float32, [None, 10])  # 标签，图像真是类别

# 定义权重、偏置
w = tf.Variable(tf.random_normal([784, 10]))  # 权重  784行10列
b = tf.Variable(tf.zeros([10]))  # 10个偏置

# 构建模型，计算预测结果
pred_y = tf.nn.softmax(tf.matmul(x, w) + b)
# 损失函数
cross_entropy = -tf.reduce_sum(y * tf.log(pred_y), reduction_indices=1)
cost = tf.reduce_mean(cross_entropy)  # 求均值
# 定义梯度下降优化器
optimizer = tf.train.GradientDescentOptimizer(0.01).minimize(cost)

batch_size = 100
saver = tf.train.Saver()

model_path = './model/mnist/mnist_model.ckpt'  # 模型保存路径

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())  # 初始化
    # 开始训练
    for epoch in range(200):
        # 计算总批次
        total_batch = int(mnist.train.num_examples / batch_size)
        avg_cost = 0.0
        for i in range(total_batch):
            # 从训练集读取一个批次的样本
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            params = {x: batch_xs, y: batch_ys}  # 参数字典

            o, c = sess.run([optimizer, cost],  # 执行的op
                            feed_dict=params)  # 喂入的参数
            avg_cost += (c / total_batch)  # 计算平均损失值
        # 每次训练结束，打印损失值
        print('epoch:%d,cost=%.9f' % (epoch + 1, avg_cost))
    print('训练结束')

    # 对模型进行评估  比较预测结果和真实结果 返回布尔类型的数组
    correct_pred = tf.equal(tf.argmax(pred_y, 1),  # 求预测结果中最大值的做因
                            tf.argmax(y, 1))  # 求真实结果中最大值索引
    # 将bool类型数组转换为浮点数，并计算准确率
    # 因为计算均值，准确率公式相同，所以调用计算机均值的函数计算准确率
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
    print('accuracy:', accuracy.eval({x: mnist.test.images,  # 测试集下的图像数据
                                      y: mnist.test.labels}))  # 测试集下图像的真实类别
    # 保存模型
    save_path = saver.save(sess, model_path)
    print('模型已保存：', save_path)

# 从测试集中随机读取两张图像，执行预测
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    saver.restore(sess, model_path)  # 加载模型

    # 从测试集中读取样本
    batch_xs, batch_ys = mnist.test.next_batch(2)
    output = tf.argmax(pred_y, 1)  # 直接取出预测结果中的最大值

    output_val, predv = sess.run([output, pred_y], # 预测op
                                 feed_dict={x: batch_xs}) # 预测，所以不需要传入标签
    print('预测最终结果：\n',output_val,'\n')
    print('真实结果：\n',batch_ys,'\n')
    print('预测的概率：\n',predv,'\n')

    # 显示图片
    im = batch_xs[0] # 第一个测试样本
    im = im.reshape(-1,28) # 28列，-1表示经过计算的值
    pylab.imshow(im)
    pylab.show()

    # 显示图片
    im = batch_xs[1] # 第2个测试样本
    im = im.reshape(-1,28) # 28列，-1表示经过计算的值
    pylab.imshow(im)
    pylab.show()

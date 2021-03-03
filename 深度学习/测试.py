# 手写体识别案例
# 模型:全连接模型
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import pylab

# 定义样本读取对象
mnist = input_data.read_data_sets('MNIST_data/',  # 数据所在目录
                                  one_hot=True)  # 标签是否采用独热编码

# 定义占位符,用于表图像数据
x = tf.placeholder(tf.float32, [None, 784])  # 图像数据,N行784列
y = tf.placeholder(tf.float32, [None, 10])  # 标签(图像真实类别),N行10列

# 定义权重,偏置
w = tf.Variable(tf.random_normal([784, 10]))  # 权重,784行10列
b = tf.Variable(tf.zeros([10]))  # 偏置,10个偏置

# 构建模型,计算预测结果
pred_y = tf.nn.softmax(tf.matmul(x, w) + b)
# 损失函数
cross_entropy = -tf.reduce_sum(y * tf.log(pred_y), reduction_indices=1)
cost = tf.reduce_mean(cross_entropy)  # 求均值
# 梯度下降优化器
optimizer = tf.train.GradientDescentOptimizer(0.01).minimize(cost)

batch_size = 100  # 批次大小
saver = tf.train.Saver()  # 模型的保存
model_path = './model/mnist/mnist_model.ckpt'  # 模型路径

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())  # 初始化
    # 开始训练
    for epoch in range(10):
        # 计算总批次
        total_batch = int(mnist.train.num_examples / batch_size)
        avg_cost = 0.0
        for i in range(total_batch):
            # 从训练集读取批次样本
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            params = {x: batch_xs, y: batch_ys}  # 参数字典
            o, c = sess.run([optimizer, cost],  # 执行op,梯度下降和损失函数
                            feed_dict=params)  # 喂入参数
            avg_cost += (c / total_batch)  # 计算评价损失值
        print('epoch:%d,cost=%.9f' % (epoch + 1, avg_cost))
    print('训练结束.')
    # 模型评估
    # 比较预测结果与真实结果,返回布尔类型的数组
    correct_pred = tf.equal(tf.argmax(pred_y, 1),  # 求预测结果中最大值索引
                            tf.argmax(y, 1))  # 求真实结果中的最大值索引
    # 将布尔类型的数组转换为浮点数,并计算准确率
    # 因为计算均值,准确率公式相同,所有调用计算均值的函数计算准确率
    accuray = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
    print('accuracy:', accuray.eval({x: mnist.test.images,  # 测试集下的图像数据
                                     y: mnist.test.labels}))  # 测试集下图像的真实类别
    # 保存模型
    save_path = saver.save(sess, model_path)
    print('模型已保存:', save_path)

# 从测试集中随机读取2张图像,执行预测
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    saver.restore(sess, model_path)  # 加载模型

    # 从测试集中读取样本
    batch_xs, batch_ys = mnist.test.next_batch(2)
    output = tf.argmax(pred_y, 1)  # 直接取出测试结果中的的最大值
    output_val, predv = sess.run([output, pred_y],  # 执行op
                                 feed_dict={x: batch_xs})  # 预测,所以不需要传入标签
    print('预测最终结果:\n', output_val, '\n')
    print('真实结果:\n', batch_ys, '\n')
    print('预测概率:\n', predv, '\n')
    # 显示图片
    im = batch_xs[0]  # 第一个测试样本
    im = im.reshape(-1, 28)  # 28列,-1表示经过计算的值
    pylab.imshow(im)
    pylab.show()

    im = batch_xs[1]  # 第二个测试样本
    im = im.reshape(-1, 28)  # 28列,-1表示经过计算的值
    pylab.imshow(im)
    pylab.show()

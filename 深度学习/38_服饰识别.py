import tensorflow as tf
from tensorflow.contrib.learn.python.learn.datasets.mnist import read_data_sets


# 定义类
class FashionMinst():
    out_features1 = 12  # 第一组卷积层输出通道数量（即第一个卷积曾卷积核数
    out_features2 = 24  # 第2组卷积层输出通道数量（即第2个卷积曾卷积核数
    con_neurons = 512  # 全连接层神经元数量

    def __init__(self, path):
        """
        构造方法
        :param path: 指定数据集的目录
        """
        self.sess = tf.Session()
        self.data = read_data_sets(path, one_hot=True)
        self.tt = None

    def init_weight_variable(self, shape):
        """
        根据指定的形状初始化权重
        :param shape: 指定要初始化的变量的形状
        :return: 返回经过初始化的变量（张量）
        """
        initial = tf.truncated_normal(shape, stddev=0.1)  # 截尾正态分布
        return tf.Variable(initial)

    def init_bias_variable(self, shape):
        """
        初始化偏置
        :param shape: 指定要初始化变量的形状
        :return: 返回经过初始化的变量（张量）
        """
        initial = tf.constant(1.0, shape=shape)
        return tf.Variable(initial)

    def conv2d(self, x, w):
        """
        二维卷积方法
        :param x: 原始数据
        :param w: 卷积核
        :return: 返回卷积运算的结果
        """
        # 卷积核：【高度，宽度，输入通道数，输出通道数】

        return tf.nn.conv2d(x, w,
                            strides=[1, 1, 1, 1],  # 各维度上的步长值
                            padding='SAME')  # 输入矩阵和输出矩阵大小一样

    def max_pool_2x2(self, x):
        """
        定义池化方法
        :param x: 原始数据
        :return: 池化计算结果
        """
        return tf.nn.max_pool(x,
                              ksize=[1, 2, 2, 1],  # 池化区域的大小
                              strides=[1, 2, 2, 1],  # 各个维度上的步长值 2
                              padding='SAME'
                              )

    def create_conv_pool_layer(self, input, input_features, out_features):
        """
        定义卷积，激活，池化层
        :param input: 原始数据
        :param input_features:输入特征数量
        :param out_features: 输出特征数量
        :return: 卷积，激活，池化层运算结果
        """
        # 卷积核
        filter = self.init_weight_variable([5, 5, input_features, out_features])  # 5,5,1,12
        b_conv = self.init_bias_variable([out_features])  # 偏置，有多少输出就有多少偏置
        h_conv = tf.nn.relu(self.conv2d(input, filter) + b_conv)  # 卷积激活运算
        h_pool = self.max_pool_2x2(h_conv)  # 2x2区域最大池化
        return h_pool

    def create_fc_layer(self, h_pool_flat, input_features, con_neurons):
        """
        创建全连接层
        :param h_pool_flat:输入数据，经过拉伸后的一维的张量
        :param input_features: 输入特征数量
        :param con_neurons: 神经元数量（输出特征的数量）
        :return: 经过全连接计算后的结果
        """
        w_fc = self.init_weight_variable([input_features, con_neurons]) # 初始化权重
        b_fc = self.init_bias_variable([con_neurons]) # 初始化偏置
        h_fc1 = tf.nn.relu(tf.matmul(h_pool_flat, w_fc) + b_fc)
        return h_fc1

    def build(self):
        """
        组建CNN卷积神经网络模型
        :return:
        """
        # 定义输入数据、标签数据的占位符
        self.x = tf.placeholder(tf.float32, shape=[None, 784])
        x_image = tf.reshape(self.x, [-1, 28, 28, 1])  # 变维-->28*28单通道的变量
        self.y_ = tf.placeholder(tf.float32, shape=[None, 10])  # 标签 N个样本 每个样本10个类别

        # 第一组卷积池化
        h_pool1 = self.create_conv_pool_layer(x_image, 1, self.out_features1)

        # 第二成
        h_pool2 = self.create_conv_pool_layer(h_pool1,
                                              self.out_features1,  # 输入特征的数量
                                              self.out_features2)  # 输出特征的数量
        # 全连接
        h_pool2_flat_features = 7 * 7 * self.out_features2  # 计算特征点的数量 1176
        h_pool2_flat = tf.reshape(h_pool2, [-1, h_pool2_flat_features])  # 拉伸成1维
        h_fc = self.create_fc_layer(h_pool2_flat,  # 输入数据
                                    h_pool2_flat_features,  # 输入特征数量
                                    self.con_neurons)  # 输出特征数量
        # dropout(通过随即丢弃一定比例神经元参数更新，防止过拟合)
        self.keep_prob = tf.placeholder('float')  # 保存率
        h_fc1_drop = tf.nn.dropout(h_fc, self.keep_prob)  # ? 10

        # 输出层
        w_fc = self.init_weight_variable([self.con_neurons, 10])  # 512行10列
        b_fc = self.init_bias_variable([10])  # 10个偏置
        y_conv = tf.matmul(h_fc1_drop, w_fc) + b_fc  # 计算wx+b

        # 计算准确率
        correct_prediction = tf.equal(tf.argmax(y_conv, 1),
                                      tf.argmax(self.y_, 1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        # 构建损失函数
        loss_func = tf.nn.softmax_cross_entropy_with_logits(labels=self.y_,  # 真实结果
                                                            logits=y_conv)  # 预测结果
        cross_entropy = tf.reduce_mean(loss_func)

        # 优化器
        optimizer = tf.train.AdagradOptimizer(0.001)  # 学习率
        self.train_step = optimizer.minimize(cross_entropy)

    def train(self):
        self.sess.run(tf.global_variables_initializer())
        batch_size = 100  # 批次大小
        print('begin trainning...')

        for i in range(10):
            total_batch = int(self.data.train.num_examples / batch_size)  # 计算批次数量
            for j in range(total_batch):
                batch = self.data.train.next_batch(batch_size)  # 获取一个批次样本
                params = {self.x: batch[0],  # 图像
                          self.y_: batch[1],  # 标签
                          self.keep_prob: 0.5}  # 计算丢弃率
                t, acc = self.sess.run([self.train_step, self.accuracy],  # 执行op
                                       params)
                if j % 100 == 0:
                    print('i: %d,j:%d , acc: %f' % (i, j, acc))

    def eval(self, x, y, keep_prob):
        params = {self.x: x, self.y_: y, self.keep_prob: 0.5}
        test_acc = self.sess.run(self.accuracy, params)  # 计算准确率
        print('Test Accuracy: %f' % test_acc)

    def close(self):
        self.sess.close()


if __name__ == '__main__':
    mnist = FashionMinst('./fashion-mnist/')
    mnist.build()  # 组建网络
    mnist.train()

    # 评估准确率
    xs, ys = mnist.data.test.next_batch(100)
    mnist.eval(xs, ys, 0.5)
    mnist.close()

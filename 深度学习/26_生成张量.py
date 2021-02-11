import tensorflow as tf

# 创建值全部为0的张量
tensor_zeros = tf.zeros(shape=[2, 3],  # 2行3列
                        dtype='float32')  # 类型

# 创建值全部为1的张量
tensor_ones = tf.ones(shape=[2, 3], dtype='float32')

# 创建正态分布随机张量
tensor_nd = tf.random_normal(shape=[10],  # 一维，10个元素
                             mean=1.7,  # 中位数
                             stddev=0.2,  # 标准差
                             dtype='float32')
# 创建形状和tensor_ones一样，值为0的张量
tensor_zeros_like = tf.zeros_like(tensor_ones)

with tf.Session() as sess:
    print(tensor_zeros.eval()) # eval表示在session中执行计算
    print(tensor_ones.eval())
    print(tensor_nd.eval())
    print(tensor_zeros_like.eval())

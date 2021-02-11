import tensorflow as tf
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 调整警告级别

a = tf.constant(5.0)
b = tf.constant(1.0)
c = tf.add(a, b)
print('c:', c)

graph = tf.get_default_graph()  # 获取缺省图
print(graph)

with tf.Session() as sess:
    print(sess.run(c))
    print(a.graph)  # 通过tensor获取graph对象
    print(c.graph)  # 通过op获取graph对象
    print(sess.graph)  # 通过session获取graph对象

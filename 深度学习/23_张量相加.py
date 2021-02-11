# 常量加法运算示例
import tensorflow as tf
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # 调整警告级别

a = tf.constant(5.0) # 定义常量a
b = tf.constant(1.0)
c = tf.add(a,b)
print('c:',c)

graph = tf.get_default_graph()
print(graph)

with tf.Session() as sess:
    print(sess.run(c))
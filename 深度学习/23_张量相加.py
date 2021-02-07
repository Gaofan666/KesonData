# 常量加法运算示例
import tensorflow.compat.v1 as tf
import os
tf.disable_v2_behavior()

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # 调整警告级别

a = tf.constant(5.0) # 定义常量a
b = tf.constant(1.0)
c = tf.add(a,b)
print('c:',c)

graph = tf.get_default_graph()
print(graph)

with tf.Session() as sess:
    print(sess.run(c))
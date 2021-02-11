# 将图中的信息存入事件文件，并在tensorboard中显示
import tensorflow as tf

# 创建一组操作
a = tf.constant([1, 2, 3, 4, 5])
var = tf.Variable(tf.random_normal([2, 3], mean=0.0, stddev=1.0), name='var')
b = tf.constant(3.0, name='a')  # 这里故意将python变量名设置为b，tensorboard变量名设置为a
c = tf.constant(4.0, name='b')
d = tf.add(b, c, name='add')

# 显式初始化
init_op = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init_op)
    # 将当前 session的graph信息写入事件文件
    fw = tf.summary.FileWriter('./Summary/', graph=sess.graph)
    print(sess.run([a,var,d]))
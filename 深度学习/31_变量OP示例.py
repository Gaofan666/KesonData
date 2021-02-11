import tensorflow as tf

# 变量是一种特殊的张量，变量中存的值是张量
# 变量可以进行持久化保存，张量不可以
# 变量使用之前，要进行显式初始化

a = tf.constant([1, 2, 3, 4])

var = tf.Variable(tf.random_normal([2, 3], mean=0.0, stddev=1.0), name='var1')
var2 = tf.Variable(tf.ones([3, 3]), name='var2')

# 定义变量 先进行全局初始化（初始化也是一个OP，需要在session的run下执行）
init_op = tf.global_variables_initializer()

# var3 = tf.Variable(tf.zeros([5,1]),name='var3') # 报错，因为变量在初始化函数之后
with tf.Session() as sess:
    sess.run(init_op)
    print(sess.run([a, var]))
    print(sess.run(var2))
    # print(sess.run(var3))

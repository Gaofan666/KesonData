# tf的helloworld程序
import tensorflow as tf

hello = tf.constant('helloworld!')  # 定义一个常量
sess = tf.Session()  # !!!  替换sess = tf.Session()
print(sess.run(hello))
sess.close()

import tensorflow as tf

plhd = tf.placeholder(tf.float32, [2, 3])
plhd2 = tf.placeholder(tf.float32, [None, 3])  # 相当于n行3列，n不确定
plhd3 = tf.placeholder(tf.float32, [None, 4])

with tf.Session() as sess:
    d = [[1, 2, 3],
         [4, 5, 6]]
    e = [[1, 2, 3, 4],
         [4, 5, 6, 9]]
    print(sess.run(plhd, feed_dict={plhd: d}))  # 通过字典传入数据
    print(sess.run(plhd2, feed_dict={plhd2: d}))  # 定义为N行3列，执行时传入2行3列
    print(sess.run(plhd3, feed_dict={plhd3: e}))

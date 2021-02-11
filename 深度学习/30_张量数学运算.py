import tensorflow as tf

x = tf.constant([[1, 2], [3, 4]], dtype=tf.float32)
y = tf.constant([[4, 3], [3, 2]], dtype=tf.float32)

x_add_y = tf.add(x, y)

x_mul_y = tf.matmul(x, y)  # 张量相乘

log_x = tf.log(x)  # 对数

x_sum_1 = tf.reduce_sum(x, axis=[1])  # 1表示行方向  0表示列方向  维度求和
x_sum_0 = tf.reduce_sum(x, axis=[0])  # 1表示行方向  0表示列方向  维度求和

# 张量计算片段和
data = tf.constant([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=tf.float32)
segment_idx = tf.constant([0, 0, 0, 1, 1, 2, 2, 2, 2, 2], dtype=tf.int32)  # 相同的数对应元素求和
x_seg_sum = tf.segment_sum(data,segment_idx)

with tf.Session() as sess:
    print(x_add_y.eval())
    print(x_mul_y.eval())
    print(log_x.eval())
    print('--------------------------')
    print(x_sum_1.eval())
    print(x_sum_0.eval())
    print(x_seg_sum.eval())

import tensorflow as tf

tensor_ones = tf.ones(shape=[2, 3], dtype='int32')
tensor_float = tf.constant([1.1, 2.2, 3.3])

with tf.Session() as sess:
    print(tf.cast(tensor_ones, tf.float32).eval())

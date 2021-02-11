import tensorflow as tf

pld = tf.placeholder(tf.float32, [None, 3])
pld.set_shape([4, 3])  # 静态形状，只能设置一次

# pld.set_shape([3,4]) #ValueError: Dimension 0 in both shapes must be equal, but are 4 and 3. Shapes are [4,3] and [3,4].

new_pld = tf.reshape([3, 4])
print(new_pld)

with tf.Session() as sess:
    pass

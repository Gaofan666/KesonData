import tensorflow as tf
a= tf.constant(5.0)

with tf.Session() as sess:
    print(sess.run(a))
    print('name:',a.name)
    print('dtype:',a.dtype)
    print('shape:',a.shape)
    print('op:',a.op)
    print('graph:',a.graph)
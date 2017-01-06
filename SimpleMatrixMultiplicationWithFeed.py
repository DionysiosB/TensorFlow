import tensorflow as tf

m1 = tf.placeholder(tf.float32)
m2 = tf.placeholder(tf.float32)
m3 = tf.matmul(m1, m2)

with tf.Session() as sess:
    mdict = {m1: [[1.0, 2.0]], m2: [[3.0],[4.0]]}
    print(sess.run(m3, mdict))

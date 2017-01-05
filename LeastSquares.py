import tensorflow as tf
import numpy as np

xd = np.random.rand(100).astype(np.float32)
g = 0.17
h = 0.31
yd = g * xd + h


w = tf.Variable(tf.random_uniform([1], -1, 1))
b = tf.Variable(tf.zeros([1]))
y = tf.add(tf.mul(w, xd), b)


init = tf.global_variables_initializer()

step = 0.5
diff = tf.reduce_mean(tf.square(y - yd))
train = tf.train.GradientDescentOptimizer(step).minimize(diff)

with tf.Session() as sess:
    sess.run(init)
    for iteration in xrange(200):
        sess.run(train)
        if(iteration % 20 == 0):
            print(iteration, sess.run(w), sess.run(b))


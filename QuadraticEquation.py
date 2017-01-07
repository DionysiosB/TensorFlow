import tensorflow as tf

x = tf.placeholder(tf.float32)
y = tf.add(tf.add(tf.mul(x, x), tf.mul(5.0, x)), 7.0)   # x^2 + 5x + 7

init = tf.global_variables_initializer()

value = 11  #Input value
with tf.Session() as sess:
    xdict = {x: value}
    sess.run(init, xdict)
    print(sess.run(y, xdict)

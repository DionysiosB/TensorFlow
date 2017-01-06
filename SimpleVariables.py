import tensorflow as tf

state = tf.Variable(0, name="counter")
one = tf.constant(1)
addone = tf.add(state, one)
update = tf.assign(state, addone)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    print(sess.run(state))
    for _ in xrange(5): print(sess.run(update))

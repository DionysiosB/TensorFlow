import tensorflow as tf

matrix1 = tf.constant([[1.0, 2.0]])
matrix2 = tf.constant([[3.0], [4.0]])
prod = tf.matmul(matrix1, matrix2)

with tf.Session() as sess: print(sess.run([prod]))

import tensorflow as tf

dtype=tf.float32

s_01 = \
[[1, 0, 0, 0],
[0, 0, -1, 0],
[1, 1, 1, 0]]

s_01 = tf.constant(s_01,dtype=dtype)

th_01 = \
[[0.041, 0, 0, 0],
[0, 0, 0.017, 0],
[0.23, 0.39, 0.23, 0]]
th_01 = tf.Variable(th_01,dtype=dtype)

s_12 =\
[[-1, 0, 1, 1],
[0, 1, -1, -1],
[-1, 1, 1, 1],
[1, -1, -1, 1]]
s_12 = tf.constant(s_12,dtype=dtype)

th_12 = \
[[0.49, 0, 0.27, 1.3],
[0, 0.23, 0.51, 0.55],
[0.28, 0.16, 0.067, 0.69],
[0.86, 1, 0.25, 0.076]]
th_12 = tf.Variable(th_12,dtype=dtype)

s_23 = \
[[1, 1, 1],
[0, 1, 0],
[1, 1, -1],
[-1, -1, 1]]
s_23 = tf.constant(s_23,dtype=dtype)

th_23 = \
[[0.13, 0.6, 0.014],
[0, 0.015, 0],
[0.26, 0.4, 0.71],
[0.39, 0.019, 0.13]]

th_23 = tf.Variable(th_23,dtype=dtype)

I = tf.constant([[1,1,1]],dtype=dtype)
label = tf.constant([[0,0,1]],dtype=dtype)

w_01 = s_01 * tf.nn.relu(th_01)
w_12 = s_12 * tf.nn.relu(th_12)
w_23 = s_23 * tf.nn.relu(th_23)

a_1 = tf.matmul(I,w_01)
a_1 = tf.nn.relu(a_1)

a_2 = tf.matmul(a_1,w_12)
a_2 = tf.nn.relu(a_2)

y = tf.matmul(a_2,w_23)

prob = tf.nn.softmax(y)

cost = tf.nn.softmax_cross_entropy_with_logits(labels=label, logits=y)

g_01 = tf.gradients(cost,th_01)
g_12 = tf.gradients(cost,th_12)
g_23 = tf.gradients(cost,th_23)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

print('a_1:')
print(sess.run(a_1))

print('a_2:')
print(sess.run(a_2))

print('Probability:')
print(sess.run(prob))


print('d3:')
print(sess.run(tf.gradients(cost,y)))

print('G 01:')
print(sess.run(g_01))

print('G 12:')
print(sess.run(g_12))

print('G 23:')
print(sess.run(g_23))




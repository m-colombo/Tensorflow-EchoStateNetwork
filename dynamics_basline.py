import Oger
import tensorflow as tf
import numpy as np
from ESN.esn_cell import ESNCell

# data
w_r = np.array([[0.03887243, -0.28983904, -0.53829223], [0.06456875,  0.0,  0.151112258], [-0.042949107, -0.48700565, -0.22361958]])
w_in = np.array([[0.3, 0.2], [-0.2, 0.01], [0.1, -0.4]])
w_bias = np.array([[ 0.2, -0.1, -0.34]])

x = np.array([[1, 0.3],[0.1, 0.4], [-1, 0.3], [-0.3, 0.4]])
states_zero = np.array([[0.0, 0.0, 0.0]])

## Manual
s = states_zero
states_m = np.array(states_zero)
for i in x:
    s = np.tanh(np.matmul(w_in, i) + np.matmul(s, w_r) + w_bias)
    states_m = np.append(states_m, s, axis=0)
states_m = states_m[1:]

## OGER
ESN_O = Oger.nodes.ReservoirNode(w_in=w_in, w_bias=w_bias, w=w_r.transpose(), output_dim=3, reset_states=True)
ESN_O.states = states_zero
states_O = ESN_O(x)
print states_O

## TF
with tf.variable_scope("rnn/ESNCell"):
    v1=tf.get_variable(initializer=w_r, name='ReservoirMatrix')
    v2=tf.get_variable(initializer=w_in.transpose(), name="InputMatrix")
    v3=tf.get_variable(initializer=w_bias[0], name="Bias")

tf.get_variable_scope().reuse_variables()
ESN_TF = ESNCell(num_units=np.size(w_r, axis=1))
(outs, _) = tf.nn.dynamic_rnn(cell=ESN_TF, inputs=np.reshape(x, [1,4,2]), initial_state=states_zero, time_major=False)

sess = tf.Session()
sess.run(tf.global_variables_initializer())
states_tf = sess.run(outs)
print states_tf

sess.close()
import tensorflow as tf
import numpy as np
from esn_cell import ESNCell


class ESNCellTest(tf.test.TestCase):

  def test_esn_dynamics(self):
    """ Simple test of reservoir dynamics """

    # Data
    w_r = np.array([[0.03887243, -0.28983904, -0.53829223],
                    [0.06456875, 0.0, 0.151112258],
                    [-0.042949107, -0.48700565, -0.22361958]])
    w_in = np.array([[0.3, 0.2], [-0.2, 0.01], [0.1, -0.4]])
    w_bias = np.array([[0.2, -0.1, -0.34]])

    x = np.array([[1, 0.3], [0.1, 0.4], [-1, 0.3], [-0.3, 0.4]])
    states_zero = np.array([[0.0, 0.0, 0.0]])

    # Manually compute reservoir states
    s = states_zero
    states_manual = np.array(states_zero)
    for i in x:
      s = np.tanh(np.matmul(w_in, i) + np.matmul(s, w_r) + w_bias)
      states_manual = np.append(states_manual, s, axis=0)
    states_manual = states_manual[1:]

    # Oger
    # ESN_O = Oger.nodes.ReservoirNode(w_in=w_in, w_bias=w_bias, w=w_r.transpose(), output_dim=3, reset_states=True)
    # ESN_O.states = states_zero
    # states_Oger = ESN_O(x)

    # Tensorflow
    with tf.variable_scope("rnn/ESNCell"):
      tf.get_variable(initializer=w_r, name='ReservoirMatrix')
      tf.get_variable(initializer=w_in.transpose(), name="InputMatrix")
      tf.get_variable(initializer=w_bias[0], name="Bias")

    tf.get_variable_scope().reuse_variables()
    cell = ESNCell(num_units=np.size(w_r, axis=1))
    (outs, _) = tf.nn.dynamic_rnn(cell=cell, inputs=np.reshape(x, [1, 4, 2]), initial_state=states_zero,
                                  time_major=False)
    with tf.Session() as sess:
      sess.run(tf.global_variables_initializer())
      states_tf = sess.run(outs)

    self.assertAllClose(states_manual, states_tf[0])

  def smoke_test(self):
    """ A simple smoke test with random initialization"""

    input_size = 4
    input_length = 4
    batch_size = 2
    n_units = 4

    cell = ESNCell(n_units)
    inputs = np.random.random([input_length, batch_size, input_size])

    state = cell.zero_state(batch_size, tf.float64)
    for i in range(input_length):
      if i > 0 : tf.get_variable_scope().reuse_variables()
      state, _ = cell(inputs[i, :, :], state)

    with self.test_session() as sess:
      sess.run(tf.global_variables_initializer())

      final_states = sess.run(state)

    expected_final_states = [[-0.56735968, -0.21625957,  0.69647415, -0.91361383],
                             [-0.22654705, -0.15751715,  0.85077971, -0.89757621]]

    self.assertAllClose(final_states, expected_final_states)

if __name__ == "__main__":
  tf.test.main()
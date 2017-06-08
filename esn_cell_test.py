import tensorflow as tf
import numpy as np
from esn_cell import ESNCell


class ESNCellTest(tf.test.TestCase):

  def test_esn_dynamics(self):
    """ A simple smoke test """

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
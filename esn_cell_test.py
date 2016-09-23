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
      sess.run(tf.initialize_all_variables())
      final_states = sess.run([state])

    expected_final_states = [[[0.75952783, -0.96463442, 0.72289173, 0.38016839],
                             [0.82451594, -0.99358452, 0.86248011, 0.24540841]]]

    self.assertAllClose(final_states, expected_final_states)

if __name__ == "__main__":
  tf.test.main()
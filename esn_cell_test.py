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

    expected_final_states = [[ 0.92921495, 0.98763743, -0.2748242, 0.71902763],
                             [ 0.99235595, 0.99932796, 0.41149584, 0.9159835 ]]

    self.assertAllClose(final_states, expected_final_states)

if __name__ == "__main__":
  tf.test.main()
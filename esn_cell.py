from tensorflow.contrib import rnn
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.framework.ops import convert_to_tensor


class ESNCell(rnn.RNNCell):
  """Echo State Network Cell.

    Based on http://www.faculty.jacobs-university.de/hjaeger/pubs/EchoStatesTechRep.pdf
    Only the reservoir, the randomized recurrent layer, is modelled. The readout trainable layer
    which map reservoir output to the target output is not implemented by this cell,
    thus neither are feedback from readout to the reservoir (a quite common technique).

    Here a practical guide to use Echo State Networks:
    http://minds.jacobs-university.de/sites/default/files/uploads/papers/PracticalESN.pdf

    Since at the moment TF doesn't provide a way to compute spectral radius
    of a matrix the echo state property necessary condition `max(eig(W)) < 1` is approximated
    scaling the norm 2 of the reservoir matrix which is an upper bound of the spectral radius.
    See https://en.wikipedia.org/wiki/Matrix_norm, the section on induced norms.

  """

  def __init__(self, num_units, wr2_scale=0.7, connectivity=0.1, leaky=1.0, activation=math_ops.tanh,
               win_init=init_ops.random_normal_initializer(),
               wr_init=init_ops.random_normal_initializer(),
               bias_init=init_ops.random_normal_initializer()):
    """Initialize the Echo State Network Cell.

    Args:
      num_units: Int or 0-D Int Tensor, the number of units in the reservoir
      wr2_scale: desired norm2 of reservoir weight matrix.
        `wr2_scale < 1` is a sufficient condition for echo state property.
      connectivity: connection probability between two reservoir units
      leaky: leaky parameter
      activation: activation function
      win_init: initializer for input weights
      wr_init: used to initialize reservoir weights before applying connectivity mask and scaling
      bias_init: initializer for biases
    """
    self._num_units = num_units
    self._leaky = leaky
    self._activation = activation

    def _wr_initializer(shape, dtype, partition_info=None):
      wr = wr_init(shape, dtype=dtype)

      connectivity_mask = math_ops.cast(
          math_ops.less_equal(
            random_ops.random_uniform(shape),
            connectivity),
        dtype)

      wr = math_ops.multiply(wr, connectivity_mask)

      wr_norm2 = math_ops.sqrt(math_ops.reduce_sum(math_ops.square(wr)))

      is_norm_0 = math_ops.cast(math_ops.equal(wr_norm2, 0), dtype)

      wr = wr * wr2_scale / (wr_norm2 + 1 * is_norm_0)

      return wr

    self._win_initializer = win_init
    self._bias_initializer = bias_init
    self._wr_initializer = _wr_initializer

  @property
  def output_size(self):
    return self._num_units

  @property
  def state_size(self):
    return self._num_units

  def __call__(self, inputs, state, scope=None):
    """ Run one step of ESN Cell

        Args:
          inputs: `2-D Tensor` with shape `[batch_size x input_size]`.
          state: `2-D Tensor` with shape `[batch_size x self.state_size]`.
          scope: VariableScope for the created subgraph; defaults to class `ESNCell`.

        Returns:
          A tuple `(output, new_state)`, computed as
          `output = new_state = (1 - leaky) * state + leaky * activation(Win * input + Wr * state + B)`.

        Raises:
          ValueError: if `inputs` or `state` tensor size mismatch the previously provided dimension.
          """

    inputs = convert_to_tensor(inputs)
    input_size = inputs.get_shape().as_list()[1]
    dtype = inputs.dtype

    with vs.variable_scope(scope or type(self).__name__):  # "ESNCell"

      win = vs.get_variable("InputMatrix", [input_size, self._num_units], dtype=dtype,
                            trainable=False, initializer=self._win_initializer)
      wr = vs.get_variable("ReservoirMatrix", [self._num_units, self._num_units], dtype=dtype,
                           trainable=False, initializer=self._wr_initializer)
      b = vs.get_variable("Bias", [self._num_units], dtype=dtype, trainable=False, initializer=self._bias_initializer)

      in_mat = array_ops.concat([inputs, state], axis=1)
      weights_mat = array_ops.concat([win, wr], axis=0)

      output = (1 - self._leaky) * state + self._leaky * self._activation(math_ops.matmul(in_mat, weights_mat) + b)

    return output, output

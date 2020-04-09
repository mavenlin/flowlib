import tensorflow as tf
import numpy as np


class ActivationNormalization(tf.keras.layers.Layer):
  ddi = tf.Variable(False, trainable=False, name="ddi")
  def __init__(self, log_scale_factor=3., scale_only=False):
    super().__init__()
    self.log_scale_factor = log_scale_factor
    self.scale_only = scale_only

  def build(self, input_shape):
    with tf.name_scope("activation_normalization"):
      channels = input_shape[-1]
      self._log_scale = tf.Variable(tf.zeros([channels]), name="log_scale")
      if not self.scale_only:
        self.shift = tf.Variable(tf.zeros([channels]), name="shift")
    super().build(input_shape)

  def data_dependent_initialize(self, x):
    if ActivationNormalization.ddi:
      x_mean, x_var = tf.nn.moments(tf.reshape(x, [-1, x.shape[-1]]), [0])
      if not self.scale_only:
        self.shift.assign(-x_mean)
        self._log_scale.assign(tf.math.log(1./(tf.sqrt(x_var) + 1e-6))
                               / self.log_scale_factor)
      else:
        pass

  @property
  def log_scale(self):
    return self._log_scale * self.log_scale_factor

  def call(self, x):
    self.data_dependent_initialize(x)
    if not self.scale_only:
      y = (x + self.shift)
    else:
      y = x
    y = y * tf.exp(self.log_scale)
    return y


class BlockDense(tf.keras.layers.Layer):
  def __init__(self, in_dim, out_dim, chunk_size, sigma=0.01):
    super().__init__()
    num_block = in_dim // chunk_size
    self.weight = tf.Variable(
      tf.random.normal([num_block, chunk_size, out_dim // num_block], 0., sigma))
    self.bias = tf.Variable(tf.zeros([num_block, 1, out_dim // num_block]))

  def call(self, x):
    y = tf.linalg.matmul(x, self.weight) + self.bias
    return y

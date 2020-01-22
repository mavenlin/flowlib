"""
Code adapted from
https://github.com/jornpeters/integer_discrete_flows/blob/master/models/backround.py
round/floor/ceil functions that support gradient
Note that floor / ceil is implemented with round, thus it may have different
behavior at integer points than the real floor ceil
"""

import tensorflow as tf


@tf.custom_gradient
def _round_straight_through(x):
  def straight_through(dy):
    return dy
  return tf.math.round(x), straight_through


def _stacked_sigmoid(x, temperature, n_approx=3):
  x_ = x - 0.5
  rounded = tf.math.round(x_)
  x_remainder = x_ - rounded
  x_remainder = tf.expand_dims(x_remainder, axis=-1)
  translation = tf.cast(tf.range(n_approx) - n_approx // 2, tf.float32)
  translation = tf.reshape(translation, [1] * x.shape.ndims + [n_approx])
  out = tf.sigmoid(x_remainder - translation) / temperature
  out = tf.reduce_sum(out, axis=-1)
  return out + rounded - (n_approx // 2)


def round(x, inverse_bin_size=256., temperature=0.2, n_approx=3):
  x = x * inverse_bin_size
  if temperature is None or temperature > 0.25:
    y = _round_straight_through(x)
  else:
    x = _stacked_sigmoid(x, temperature, n_approx)
    y = _round_straight_through(x)
  y = y / inverse_bin_size
  return y


def floor(x, temperature=None, n_approx=3):
  return round(x - 0.5)


def ceil(x, temperature=None, n_approx=3):
  return round(x + 0.5)

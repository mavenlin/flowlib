"""bijector on discrete valued tensors
"""

import tensorflow as tf
import edward2 as ed
import numpy as np
from .real import Bijector, SequentialBijector, BijectorIO, copy_on_write
from .round import round


class SplitFactorCoupling(Bijector):
  def __init__(self, in_channels, size_1, make_network):
    super().__init__()
    self.in_channels = in_channels
    self.size_1 = size_1
    self.size_2 = in_channels - size_1
    self.f = make_network(self.size_1, self.size_2)

  def call(self, input, inverse=False):
    x = input.sample
    x1, x2 = tf.split(x, [self.size_1, self.size_2], axis=-1)
    shift = self.f(x1)
    rshift = round(shift)
    if not inverse:
      x2 = x2 + rshift
    else:
      x2 = x2 - rshift
    x = tf.concat([x1, x2], axis=-1)
    return copy_on_write(input, sample=x)


class EdwardBijector(Bijector):
  def __init__(self, edward_flow):
    self.edward_flow = edward_flow

  def call(self, input, inverse=False):
    x = input.sample
    shape = x.shape
    x = tf.reshape(x, [-1, x.shape[-2], x.shape[-1]])
    if not inverse:
      y = self.edward_flow(x)
    else:
      y = self.edward_flow.reverse(x)
    y = tf.reshape(y, shape)
    return copy_on_write(input, sample=y)


def make_discrete_bijector(dim, num_category, depth):
  flow = []
  for _ in range(depth):
    layer = tf.keras.Sequential()
    layer.add(tf.keras.layers.Reshape([dim * num_category]))
    layer.add(tf.keras.layers.Dense(dim * num_category, activation=tf.nn.relu))
    layer.add(tf.keras.layers.Dense(dim * num_category, activation=tf.nn.relu))
    layer.add(tf.keras.layers.Dense(dim * num_category))
    layer.add(tf.keras.layers.Reshape([dim, num_category]))
    if depth % 2 == 0:
      mask = tf.concat([tf.ones([dim // 2]), tf.zeros([dim // 2])],
                       axis=0)
    else:
      mask = tf.concat([tf.zeros([dim // 2]), tf.ones([dim // 2])],
                       axis=0)
    f = ed.layers.DiscreteBipartiteFlow(layer=layer, mask=mask, temperature=1.)
    flow.append(EdwardBijector(f))

  flow = SequentialBijector(flow)
  return flow

if __name__ == "__main__":
  dim = 100
  flow = make_discrete_bijector(dim, depth=3)
  target = ed.OneHotCategorical(
        logits=tf.Variable(tf.random.normal([dim, 2])))

  input = tf.one_hot(tf.cast(
      tf.random.normal([128, dim]) > 0., tf.int32), depth=2)
  input = tf.cast(input, tf.float32)
  outputs = flow(BijectorIO(input), inverse=True)

  with tf.GradientTape() as tape:
    tape.watch(outputs.sample)
    print(outputs.sample)
    logprob = target.distribution.log_prob(outputs.sample)
    logprob = tf.reduce_mean(logprob)
  grads = tape.gradient(logprob, outputs.sample)
  print(grads)

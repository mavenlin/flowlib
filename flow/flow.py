import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions
import numpy as np
from .real import Bijector, BijectorIO, Shaping, SequentialBijector, Flat


class UndoShape(Bijector):
  def __init__(self, sequential_bijector):
    super().__init__()
    from collections import deque
    q = deque(sequential_bijector.layers)
    reshape_ops = []
    while len(q) > 0:
      e = q.popleft()
      if isinstance(e, SequentialBijector):
        q.extendleft(reversed(e.layers))
      elif isinstance(e, Shaping):
        reshape_ops.append(e)
      else:
        continue
    self.reshape_ops = SequentialBijector(reshape_ops)

  def call(self, input, inverse=False):
    if inverse:
      return self.reshape_ops(input)
    else:
      return self.reshape_ops(input, inverse=True)


def _undo_shape(bijector):
  assert not hasattr(bijector, "_finalized")
  if not isinstance(bijector, SequentialBijector):
    bijector = SequentialBijector([bijector])
  undo_shape = UndoShape(bijector)
  bijector.layers.append(undo_shape)
  bijector._finalized = True
  return bijector


class InjectionFlow(tf.Module):
  def __init__(self, data_shape, injector, train_z_distribution=False):
    super().__init__()
    self.data_shape = data_shape
    self.injector = injector
    self.injector.layers.append(Flat())

  def distributions(self):
    return tfd.Normal(loc=0., scale=1.)

  def log_prob(self, x):
    input = BijectorIO(x, tf.zeros([x.shape[0]]), [])
    output = self.injector(input)
    self.z_shape = output.sample.shape[1:]
    logprob = self.distributions().log_prob(output.sample)
    logprob = tf.reshape(logprob, [logprob.shape[0], -1])
    logprob = tf.reduce_sum(logprob, 1)
    logprob += output.logdet
    return logprob

  def logpz(self, x):
    input = BijectorIO(x, tf.zeros([x.shape[0]]), [])
    output = self.injector(input)
    logprob = self.distributions().log_prob(output.sample)
    logprob = tf.reshape(logprob, [logprob.shape[0], -1])
    logprob = tf.reduce_mean(logprob, 1)
    return logprob

  def sample(self, batch_size):
    z = self.distributions().sample([batch_size, *self.z_shape])
    gen = self.injector(
        BijectorIO(z, tf.zeros([z.shape[0]]), []), inverse=True)
    return gen.sample, gen.logdet


class NormalizingFlow(tf.Module):
  def __init__(self, data_shape, bijector, train_z_distribution=False):
    super().__init__()
    self.data_shape = data_shape
    self.bijector = _undo_shape(bijector)
    self.z_loc = tf.Variable(tf.zeros(data_shape), name="loc",
                             trainable=train_z_distribution)
    self.z_log_scale = tf.Variable(
        tf.zeros(data_shape), name="log_scale", trainable=train_z_distribution)

  def distribution(self, temperature=1.):
    std_scale = tf.sqrt(temperature)
    return tfd.Normal(loc=self.z_loc,
                      scale=tf.exp(self.z_log_scale) * std_scale)

  def log_prob(self, x, cond=None, temperature=1.):
    input = BijectorIO(x, tf.zeros([x.shape[0]]), [], cond=cond)
    output = self.bijector(input)
    logprob = self.distribution(temperature).log_prob(output.sample)
    logprob = tf.reshape(logprob, [logprob.shape[0], -1])
    logprob = tf.reduce_sum(logprob, 1)
    logprob += output.logdet
    return logprob

  def bits_per_dim(self, x, cond=None, temperature=1., volume_per_dim=1./256.,
                   keep_batch=False):
    """compute bits per dimension for input x
    By default, the volume is set to 1./256., which means x is image data,
    and is normalized to [0, 1] for each pixel.
    """
    data_shape = x.shape[1:] # remove batch dim
    log_prob = self.log_prob(x, cond, temperature)
    if not keep_batch:
      log_prob = tf.reduce_mean(log_prob)
    log_prob_per_dim = log_prob / data_shape.num_elements()
    bits_per_dim = - (log_prob_per_dim +
                      tf.math.log(volume_per_dim)) / tf.math.log(2.)
    return bits_per_dim

  def sample(self, batch_size, cond=None, temperature=1.):
    z = self.distribution(temperature).sample(batch_size)
    gen = self.bijector(
        BijectorIO(z, tf.zeros([z.shape[0]]), [], cond=cond), inverse=True)
    return gen.sample, gen.logdet

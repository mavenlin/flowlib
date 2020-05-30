import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions
import numpy as np
from .real import Bijector, BijectorIO, Shaping, \
  SequentialBijector, Flat, FactorOut


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
    input = BijectorIO(x, tf.zeros([tf.shape(x)[0]]), [])
    output = self.injector(input)
    self.z_shape = output.sample.shape[1:]
    logprob = self.distributions().log_prob(output.sample)
    logprob = tf.reshape(logprob, [tf.shape(logprob)[0], -1])
    logprob = tf.reduce_sum(logprob, 1)
    logprob += output.logdet
    return logprob

  def logpz(self, x):
    input = BijectorIO(x, tf.zeros([tf.shape(x)[0]]), [])
    output = self.injector(input)
    logprob = self.distributions().log_prob(output.sample)
    logprob = tf.reshape(logprob, [tf.shape(logprob)[0], -1])
    logprob = tf.reduce_mean(logprob, 1)
    return logprob

  def sample(self, batch_size):
    z = self.distributions().sample([batch_size, *self.z_shape])
    gen = self.injector(
        BijectorIO(z, tf.zeros([tf.shape(z)[0]]), []), inverse=True)
    return gen.sample, gen.logdet


class NormalizingFlow(tf.Module):
  def __init__(self, data_shape, bijector, train_z_distribution=False):
    super().__init__()
    self.data_shape = data_shape
    self._bijector = bijector
    self._undo_shape = UndoShape(bijector)
    self.bijector = SequentialBijector([bijector, self._undo_shape])
    self.z_loc = tf.Variable(tf.zeros(data_shape), name="loc",
                             trainable=train_z_distribution)
    self.z_log_scale = tf.Variable(
        tf.zeros(data_shape), name="log_scale", trainable=train_z_distribution)

  def distribution(self, temperature=1.):
    std_scale = tf.sqrt(temperature)
    std_scale = tf.reshape(std_scale, [-1] + [1] * len(self.data_shape))
    return tfd.Normal(loc=self.z_loc,
                      scale=tf.exp(tf.expand_dims(self.z_log_scale, axis=0)
                      ) * std_scale)

  def decomposed_distributions(self, temperature=1.):
    std_scale = tf.sqrt(temperature)
    std_scale = tf.reshape(std_scale, [-1] + [1] * len(self.data_shape))
    scale = tf.exp(tf.expand_dims(self.z_log_scale, axis=0)) * std_scale
    loc = tf.expand_dims(self.z_loc, axis=0)
    scales = self._undo_shape(BijectorIO(sample=scale), inverse=True)
    locs = self._undo_shape(BijectorIO(sample=loc), inverse=True)
    scales = [scales.sample] + scales.facout
    locs = [locs.sample] + locs.facout
    return [tfd.Normal(loc=loc, scale=scale)
            for loc, scale in zip(locs, scales)]

  def log_prob(self, x, cond=None, temperature=1., decompose=False):
    track = ["delta_logdet"] if decompose else []
    input = BijectorIO(x, tf.zeros([tf.shape(x)[0]]), [],
                       cond=cond, track=track)
    temperature = tf.broadcast_to(temperature, [tf.shape(x)[0]])
    if not decompose:
      output = self.bijector(input)
      logprob = self.distribution(temperature).log_prob(output.sample)
      logprob = tf.reshape(logprob, [tf.shape(logprob)[0], -1])
      logprob = tf.reduce_sum(logprob, 1)
      logprob += output.logdet
      return logprob
    else:
      dists = self.decomposed_distributions(temperature)
      output = self._bijector(input)
      logdet_dict = output.track["delta_logdet"]
      logdets = []
      for bj, logdet in logdet_dict:
        if logdet is not None:
          logdets.append(logdet)
      outputs = [output.sample] + output.facout
      logprobs = [tf.reduce_sum(tf.reshape(
          dist.log_prob(output), [tf.shape(output)[0], -1]), 1)
                  for output, dist in zip(outputs, dists)]
      return logprobs, logdets


  def _logprob_to_bpd(self, log_prob, data_shape, volume_per_dim):
    log_prob_per_dim = log_prob / data_shape.num_elements()
    bits_per_dim = - (log_prob_per_dim + tf.math.log(volume_per_dim)) / tf.math.log(2.)
    return bits_per_dim

  def bits_per_dim(self, x, cond=None, temperature=1., volume_per_dim=1./256.,
                   keep_batch=False, decompose=False):
    """compute bits per dimension for input x
    By default, the volume is set to 1./256., which means x is image data,
    and is normalized to [0, 1] for each pixel.
    """
    data_shape = x.shape[1:] # remove batch dim
    bpd = lambda logprob: self._logprob_to_bpd(
        logprob, data_shape, volume_per_dim)
    if decompose:
      logprobs, logdets = self.log_prob(x, cond, temperature, decompose=True)
      if not keep_batch:
        logprobs = [tf.reduce_mean(logprob) for logprob in logprobs]
        logdets = [tf.reduce_mean(logdet) for logdet in logdets]
      bpd_z = [bpd(logprob) for logprob in logprobs]
      bpd_det = [bpd(logdet) for logdet in logdets]
      return bpd_z, bpd_det
    else:
      log_prob = self.log_prob(x, cond, temperature, decompose=False)
      if not keep_batch:
        log_prob = tf.reduce_mean(log_prob)
      return bpd(log_prob)

  def sample(self, batch_size=1, cond=None, temperature=1.):
    temperature = tf.broadcast_to(temperature, [batch_size])
    z = self.distribution(temperature).sample()
    gen = self.bijector(
        BijectorIO(z, tf.zeros([tf.shape(z)[0]]), [], cond=cond), inverse=True)
    return gen.sample

  def entropy(self):
    base_ent = self.distribution().entropy()
    sample = self.sample(16)
    output = self.bijector(BijectorIO(sample=sample, logdet=tf.zeros([16])))
    return tf.reduce_sum(base_ent) + tf.reduce_mean(output.logdet, keepdims=True)

  def cross_entropy(self, other):
    sample = self.sample(16)
    logprob = other.log_prob(sample)
    logprob = tf.reduce_mean(logprob, keepdims=True)
    return - logprob

  def kl_divergence(self, other):
    return self.cross_entropy(other) - self.entropy()

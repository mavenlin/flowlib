import tensorflow as tf
import numpy as np
import copy


clip_value = 0.


class BijectorIO(object):
  def __init__(self, sample, logdet=None, facout=[], cond=None, track=[]):
    if logdet is None:
      logdet = tf.zeros([sample.shape[0]])
    self.sample = sample
    self.logdet = logdet
    self.facout = facout
    self.cond = cond
    self.track = { k: [] for k in track }


def copy_on_write(bijector_io, **kwargs):
  new_bijector_io = copy.copy(bijector_io)
  for k, v in kwargs.items():
    setattr(new_bijector_io, k, v)
  return new_bijector_io


class Bijector(tf.Module):
  debug = False
  def __init__(self):
    super().__init__()

  def _process_log_scale(self, log_scale):
    if clip_value != 0:
      log_scale = tf.clip_by_value(log_scale, -clip_value, clip_value)
    return log_scale

  def __call__(self, input, inverse=False):
    output = self.call(input, inverse=inverse)
    if "delta_logdet" in input.track:
      if output.logdet is input.logdet:
        input.track["delta_logdet"].append((self, None))
      else:
        input.track["delta_logdet"].append((self, output.logdet - input.logdet))
    if Bijector.debug:
      tf.debugging.check_numerics(output.sample, f"{type(self)}")
    return output

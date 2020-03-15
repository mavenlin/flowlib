"""bijector on real value tensors
"""
import tensorflow as tf
import numpy as np
import copy
from .base import *
from .layers import ActivationNormalization


class Shaping(Bijector):
  """Base class of bijectors that only shapes the input without
  performing computation.
  """
  def __init__(self):
    super().__init__()


class Flat(Bijector):
  def __init__(self):
    super().__init__()

  def call(self, input, inverse=False):
    if not inverse:
      ls = [input.sample] + input.facout
      self.shape_info = [x.shape[1:] for x in ls]
      ys = [tf.reshape(x, [tf.shape(x)[0], -1]) for x in ls]
      y = tf.concat(ys, axis=-1)
      return copy_on_write(input, sample=y, facout=[])
    else:
      xs = tf.split(input.sample, [s.num_elements() for s in self.shape_info],
                    axis=-1)
      xs = [tf.reshape(x, [-1, *s]) for x, s in zip(xs, self.shape_info)]
      return copy_on_write(input, sample=xs[0], facout=xs[1:])


class Augment(Bijector):
  def __init__(self, num_channels):
    super().__init__()
    self.num_channels = num_channels
    self.network = tf.keras.Sequential()
    self.network.add(tf.keras.layers.Conv2D(num_channels, kernel_size=(1, 1),
                                            activation=tf.nn.relu))
    self.network.add(tf.keras.layers.Conv2D(num_channels, kernel_size=(1, 1),
                                            activation=tf.sigmoid))

  def call(self, input, inverse=False):
    if not inverse:
      x = input.sample
      y = self.network(x)
      sample = tf.concat([input.sample, y], axis=-1)
      return copy_on_write(input, sample=sample)
    else:
      sample = input.sample[..., :input.sample.shape[-1] - self.num_channels]
      return copy_on_write(input, sample=sample)


class ActNorm(Bijector):
  def __init__(self, channels):
    super().__init__()
    with tf.name_scope("act_norm"):
      self.act_norm = ActivationNormalization()
      self.act_norm.build([channels])

  def call(self, input, inverse=False):
    x, logdet = input.sample, input.logdet
    spatial_size = tf.cast(tf.reduce_prod(x.shape[1:-1]), tf.float32)
    self.act_norm.data_dependent_initialize(x)
    log_scale = self._process_log_scale(self.act_norm.log_scale)
    scale = tf.exp(log_scale)
    if not inverse:
      y = (x + self.act_norm.shift) * scale
      logdet += tf.reduce_sum(log_scale) * spatial_size
    else:
      y = x / scale - self.act_norm.shift
      logdet -= tf.reduce_sum(log_scale) * spatial_size
    return BijectorIO(y, logdet, input.facout)


class Permute(Bijector):
  def __init__(self, in_channels, axis=-1):
    super().__init__()
    perm = np.random.permutation(in_channels)
    inv_perm = np.arange(len(perm))[np.argsort(perm)]
    self.perm = tf.constant(perm, dtype=tf.int32)
    self.inv_perm = tf.constant(inv_perm, dtype=tf.int32)
    self.axis = axis

  def call(self, input, inverse=False):
    x = input.sample
    if not inverse:
      x = tf.gather(x, self.perm, axis=self.axis)
    else:
      x = tf.gather(x, self.inv_perm, axis=self.axis)
    return copy_on_write(input, sample=x)


class Flip(Bijector):
  def __init__(self):
    super().__init__()

  def call(self, input, inverse=False):
    x, logdet = input.sample, input.logdet
    in_channels = x.shape[-1]
    if not inverse:
      size_1, size_2 = in_channels // 2, in_channels - in_channels // 2
    else:
      size_1, size_2 = in_channels - in_channels // 2, in_channels // 2
    x1, x2 = tf.split(x, [size_1, size_2], axis=-1)
    x = tf.concat([x2, x1], axis=-1)
    return copy_on_write(input, sample=x)


class SpatialToBatch(Bijector):
  def __init__(self, spatial_size=None):
    super().__init__()
    self.spatial_size = spatial_size

  def call(self, input, inverse=False):
    if inverse:
      if self.spatial_size is None:
        raise RuntimeError("forward first then inverse")
      else:
        sample = tf.reshape(
            input.sample, [-1, *self.spatial_size, input.sample.shape[-1]])
        logdet = tf.reshape(input.logdet, [-1, *self.spatial_size])
        logdet = tf.reduce_sum(logdet, axis=[1, 2])
    else:
      if self.spatial_size is None:
        self.spatial_size = input.sample.shape[1:-1]
      sample = tf.reshape(input.sample, [-1, input.sample.shape[-1]])
      spatial_dim = tf.reduce_prod(
          tf.cast(tf.shape(input.sample)[1:-1], tf.float32))
      logdet = tf.tile(input.logdet / spatial_dim, [spatial_dim])
    return BijectorIO(sample, logdet, input.facout)


class BatchToSpatial(Bijector):
  def __init__(self, spatial_size):
    super().__init__()
    self.spatial_size = spatial_size

  def call(self, input, inverse=False):
    if not inverse:
      sample = tf.reshape(
          input.sample, [-1, *self.spatial_size, input.sample.shape[-1]])
      logdet = tf.reshape(input.logdet, [-1, *self.spatial_size])
      logdet = tf.reduce_sum(logdet, axis=[1, 2])
      return BijectorIO(sample, logdet, input.facout)
    else:
      raise RuntimeError("Not Implemented")


class ConditionalLinear(Bijector):
  def __init__(self, in_dim, cond_dim, network):
    super().__init__()
    self.in_dim = in_dim
    self.cond_dim = cond_dim
    self.f = network

  def call(self, input, inverse=False):
    x, logdet, cond = input.sample, input.logdet, input.cond
    log_scale, shift = tf.split(self.f(cond), 2, axis=-1)
    scale = tf.nn.softplus(log_scale)
    if not inverse:
      x = (x + shift) * scale
    else:
      x = x / scale - shift
    logdet += tf.math.log(tf.reduce_sum(
        tf.reshape(scale, [scale.shape[0], -1]), axis=1))
    return copy_on_write(input, sample=x, logdet=logdet)


class Coupling(Bijector):
  def __init__(self, in_channels, network):
    super().__init__()
    self.in_channels = in_channels
    self.f = network

  def call(self, input, inverse=False):
    x, logdet = input.sample, input.logdet
    size_1, size_2 = self.in_channels // 2, self.in_channels - self.in_channels // 2
    x1, x2 = tf.split(x, [size_1, size_2], axis=-1)
    log_scale, shift = tf.split(self.f(x1), 2, axis=-1)
    log_scale = self._process_log_scale(log_scale)
    scale = tf.nn.sigmoid(log_scale + 2.)

    if not inverse:
      x2 = (x2 + shift) * scale
      scale = tf.reshape(scale, [tf.shape(scale)[0], -1])
      logdet += tf.reduce_sum(tf.math.log(scale), axis=-1)
    else:
      x2 = x2 / scale - shift
      scale = tf.reshape(scale, [tf.shape(scale)[0], -1])
      logdet -= tf.reduce_sum(tf.math.log(scale), axis=-1)
    x = tf.concat([x1, x2], axis=-1)
    return copy_on_write(input, sample=x, logdet=logdet)


class Invertible1x1Conv2D(Bijector):
  def __init__(self, channels, method="svd"):
    super().__init__()
    self.method = method
    self.channels = channels
    if method == "svd":
      self.svd_init(channels)
    else:
      self.lu_init(channels)

  def lu_init(self, channels):
    self.w_shape = [channels, channels]
    import scipy
    np_w = scipy.linalg.qr(np.random.randn(*self.w_shape))[0]
    np_p, np_l, np_u = scipy.linalg.lu(np_w)
    np_s = np.diag(np_u)
    np_sign_s = np.sign(np_s)
    np_log_s = np.log(abs(np_s))
    np_u = np.triu(np_u, k=1)
    with tf.name_scope("conv1x1_lu"):
      self.p = tf.Variable(np_p, trainable=False, dtype=tf.float64, name="p")
      self.p_inv = tf.Variable(np.linalg.inv(np_p),
                               trainable=False, dtype=tf.float64, name="p_inv")
      self.l = tf.Variable(np_l, dtype=tf.float64, name="l")
      self.sign_s = tf.Variable(np_sign_s, trainable=False,
                                dtype=tf.float64, name="sign_s")
      self.log_s = tf.Variable(np_log_s, dtype=tf.float64, name="log_s")
      self.u = tf.Variable(np_u, dtype=tf.float64, name="u")
      self.l_mask = tf.constant(np.tril(
          np.ones(self.w_shape, dtype="float32"), -1), dtype=tf.float64)

  def lu_call(self, x, logdet, spatial_size, inverse=False):
    l = self.l * self.l_mask + tf.eye(*self.w_shape, dtype=tf.float64)
    log_s = self._process_log_scale(log_s)
    u = self.u * tf.transpose(self.l_mask) + tf.linalg.diag(
        self.sign_s * tf.exp(log_s))
    if not inverse:
      w = tf.matmul(self.p, tf.matmul(l, u))
      x = tf.matmul(x, tf.cast(w, tf.float32))
      logdet += tf.cast(tf.reduce_sum(log_s), tf.float32) * spatial_size
    else:
      u_inv = tf.linalg.inv(u)
      l_inv = tf.linalg.inv(l)
      p_inv = self.p_inv
      w_inv = tf.matmul(u_inv, tf.matmul(l_inv, p_inv))
      x = tf.matmul(x, tf.cast(w_inv, tf.float32))
      logdet -= tf.cast(tf.reduce_sum(log_s), tf.float32) * spatial_size
    return x, logdet

  def svd_init(self, channels):
    self.channels = channels
    with tf.name_scope("conv1x1_svd"):
      self.log_s = tf.Variable(tf.zeros([channels]), name="log_s")
      self.I = tf.constant(tf.eye(channels, batch_shape=[2]))
      self.v = tf.Variable(
          tf.random.normal([channels * 2, channels, 1]))

  def svd_call(self, x, logdet, spatial_size, inverse=False):
    # Householder transformations
    channels = self.channels
    log_s = self.log_s
    log_s = self._process_log_scale(log_s)
    s = 2 * tf.sigmoid(log_s) + 1e-6
    log_s = tf.math.log(s)
    v = self.v
    u = 2 * tf.matmul(v, v, transpose_b=True) / tf.matmul(v, v, transpose_a=True)
    u = tf.reshape(u, [channels, 2, channels, channels])
    Q = tf.expand_dims(self.I, axis=0) - u
    q = self.I
    for i in range(self.channels):
      q = tf.matmul(q, Q[i])
    if not inverse:
      u, v = q[0], q[1]
      w = tf.matmul(u * tf.reshape(s, [1, channels]), v)
      x = tf.matmul(x, w)
      logdet += tf.reduce_sum(log_s) * spatial_size
    else:
      q_inv = tf.transpose(q, [0, 2, 1])
      u_inv, v_inv = q_inv[0], q_inv[1]
      w_inv = tf.matmul(v_inv / tf.reshape(s, [1, channels]), u_inv)
      x = tf.matmul(x, w_inv)
      logdet -= tf.reduce_sum(log_s) * spatial_size
    return x, logdet

  def call(self, input, inverse=False):
    x, logdet = input.sample, input.logdet
    spatial_size = tf.cast(tf.reduce_prod(x.shape[1:-1]), tf.float32)
    y = tf.reshape(x, [-1, x.shape[-1]])
    if self.method == "svd":
      y, logdet = self.svd_call(y, logdet, spatial_size, inverse)
    else:
      y, logdet = self.lu_call(y, logdet, spatial_size, inverse)
    y = tf.reshape(y, x.shape)
    return copy_on_write(input, sample=y, logdet=logdet)


class FactorOut(Shaping):
  def __init__(self, keep_size, axis=-1):
    super().__init__()
    self.keep_size = keep_size
    self.axis = axis

  def call(self, input, inverse=False):
    x = input.sample
    if not inverse:
      out_size = x.shape[self.axis] - self.keep_size
      x1, x2 = tf.split(x, [self.keep_size, out_size], axis=self.axis)
      return copy_on_write(input, sample=x1, facout=[x2] + input.facout)
    else:
      x = tf.concat([x, input.facout[0]], axis=self.axis)
      return copy_on_write(input, sample=x, facout=input.facout[1:])


@tf.function
def reverse_extract_patches(patches, sizes, strides, rates):
  size = np.prod(sizes)
  c = patches.shape[-1] // size
  z = tf.zeros_like(patches[..., :c])
  p = tf.image.extract_patches(z, sizes, strides, rates, padding="SAME")
  summed = tf.gradients(p, z, grad_ys=patches)
  multiple = tf.gradients(p, z, grad_ys=tf.ones_like(patches))
  averaged = summed[0] / multiple[0]
  return averaged

class Im2Col(Bijector):
  """The slow way, but easier
  Note this operator is not bijective, but instead injective
  But this doesn't hurt the computation of bits per dim
  """
  def __init__(self, patch_size=[3, 3], strides=[1, 1]):
    super().__init__()
    self.patch_size = patch_size
    self.strides = strides

  def call(self, input, inverse=False):
    x = input.sample
    if not inverse:
      x= tf.image.extract_patches(
          x, [1, *patch_size, 1], [1, 1, 1, 1], [1, 1, 1, 1],
          padding='SAME')
    else: # this is not possible, but for visualization, we compute the average
      x = reverse_extract_patches(
          x, [1, *patch_size, 1], [1, 1, 1, 1], [1, 1, 1, 1])
    return copy_on_write(input, sample=x)


class ConcatZero(Bijector):
  """This is also an injector
  """
  def __init__(self, axis=-1):
    super().__init__()
    self.axis = axis

  def call(self, input, inverse=False):
    x = input.sample
    if not inverse:
      x = tf.concat([x, tf.zeros_like(x)], axis=self.axis)
    else:
      x, _ = tf.split(x, 2, axis=self.axis)
    return copy_on_write(input, sample=x)


class Squeeze2D(Shaping):
  def __init__(self, factor=2):
    super().__init__()
    self.factor = factor

  def call(self, input, inverse=False):
    x = input.sample
    factor = self.factor
    if not inverse:
      shape = x.shape
      height = shape[1]
      width = shape[2]
      n_channels = shape[3]
      x = tf.reshape(x, [-1, height // factor, factor,
                         width // factor, factor, n_channels, *shape[4:]])
      x = tf.transpose(x, [0, 1, 3, 5, 2, 4, *range(6, shape.ndims + 2, 1)])
      x = tf.reshape(x, [-1, height // factor, width // factor,
                         n_channels * factor * factor, *shape[4:]])
    else:
      shape = x.shape
      height = shape[1]
      width = shape[2]
      n_channels = shape[3]
      x = tf.reshape(x, [-1, height, width,
                         n_channels // factor ** 2, factor, factor, *shape[4:]])
      x = tf.transpose(x, [0, 1, 4, 2, 5, 3, *range(6, shape.ndims + 2, 1)])
      x = tf.reshape(x, [-1, height * factor,
                         width * factor, n_channels // factor ** 2, *shape[4:]])
    return copy_on_write(input, sample=x)


class SequentialBijector(Bijector):
  def __init__(self, layers):
    super().__init__()
    self.layers = layers

  def call(self, input, inverse=False):
    output = input
    if not inverse:
      for l in self.layers:
        output = l(output, inverse)
    else:
      for l in reversed(self.layers):
        output = l(output, inverse)
    return output


class Sigmoid(Bijector):
  def __init__(self):
    super().__init__()

  def call(self, input, inverse=False):
    x, logdet = input.sample, input.logdet
    if not inverse:
      y = tf.sigmoid(x)
      delta_logdet = tf.math.log(y - y * y)
    else:
      y = tf.math.log(x) - tf.math.log(1 - x)
      delta_logdet = - tf.math.log(x - x * x)
    delta_logdet = tf.reduce_sum(
        tf.reshape(delta_logdet, [delta_logdet.shape[0], -1]), axis=1)
    logdet += delta_logdet
    return copy_on_write(input, sample=y, logdet=logdet)


class LogitTransform(Bijector):
  def __init__(self, alpha=1e-6):
    super().__init__()
    self.alpha = alpha

  def call(self, input, inverse=False):
    x, logdet = input.sample, input.logdet

    if not inverse:
      s = self.alpha + (1. - 2. * self.alpha) * x
      y = tf.math.log(s) - tf.math.log(1 - s)
      delta_logdet = - tf.math.log(s - s * s) + tf.math.log(1. - 2. * self.alpha)
    else:
      y = (tf.sigmoid(x) - self.alpha) / (1. - 2. * self.alpha)
      s = self.alpha + (1. - 2. * self.alpha) * y
      delta_logdet = tf.math.log(s - s * s) - tf.math.log(1. - 2. * self.alpha)

    delta_logdet = tf.reduce_sum(tf.reshape(
        delta_logdet, [tf.shape(x)[0], -1]), axis=-1)
    logdet += delta_logdet
    return copy_on_write(input, sample=y, logdet=logdet)


class Dequantizer(Bijector):
  def __init__(self, dequant_bijector):
    super().__init__()
    self.bijector = dequant_bijector
    self.noise_distribution = tfd.Normal(loc=0., scale=1.)

  def call(self, input, inverse=False):
    x, logdet = input.sample, input.logdet
    epsilon = self.noise_distribution.sample(tf.shape(x))
    logp_epsilon = self.noise_distribution.log_prob(epsilon)
    logp_epsilon = tf.reduce_sum(
        tf.reshape(logp_epsilon, [tf.shape(logp_epsilon)[0], -1]), axis=1)
    dequant_i = BijectorIO(sample=epsilon, logdet=logdet, facout=[], cond=x)
    dequant_o = self.bijector(dequant_i)
    return copy_on_write(dequant_o, logdet=dequant_o.logdet - logp_epsilon)

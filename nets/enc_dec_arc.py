
# Copyright 2022 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Any
import functools

import ml_collections

import jax
import jax.numpy as jnp
import flax.linen as nn


def tensorflow_style_avg_pooling(x, window_shape, strides, padding: str):
  """Avg pooling as done by TF (Flax layer gives different results).

  To be specific, Flax includes padding cells when taking the average,
  while TF does not.

  Args:
    x: Input tensor
    window_shape: Shape of pooling window; if 1-dim tuple is just 1d pooling, if
      2-dim tuple one gets 2d pooling.
    strides: Must have the same dimension as the window_shape.
    padding: Either 'SAME' or 'VALID' to indicate pooling method.

  Returns:
    pooled: Tensor after applying pooling.
  """
  pool_sum = jax.lax.reduce_window(x, 0.0, jax.lax.add,
                                   (1,) + window_shape + (1,),
                                   (1,) + strides + (1,), padding)
  pool_denom = jax.lax.reduce_window(
      jnp.ones_like(x), 0.0, jax.lax.add, (1,) + window_shape + (1,),
      (1,) + strides + (1,), padding)
  return pool_sum / pool_denom


def dsample(x):
  return tensorflow_style_avg_pooling(x, (2, 2), strides=(2, 2), padding='same')


def upsample(x, factor=2):
  n, h, w, c = x.shape
  x = jax.image.resize(x, (n, h * factor, w * factor, c), method='nearest')
  return x


def get_norm_layer(train, dtype, norm_type='BN'):
  """Normalization layer."""
  if norm_type == 'BN':
    norm_fn = functools.partial(
        nn.BatchNorm,
        use_running_average=not train,
        momentum=0.9,
        epsilon=1e-5,
        axis_name=None,
        axis_index_groups=None,
        dtype=jnp.float32)
  elif norm_type == 'LN':
    norm_fn = functools.partial(nn.LayerNorm, dtype=dtype)
  elif norm_type == 'GN':
    norm_fn = functools.partial(nn.GroupNorm, dtype=dtype)
  else:
    raise NotImplementedError
  return norm_fn


class Downsample(nn.Module):
  """Downsample Blocks."""

  use_conv: bool

  @nn.compact
  def __call__(self, x):
    out_dim = x.shape[-1]
    if self.use_conv:
      x = nn.Conv(out_dim, kernel_size=(4, 4), strides=(2, 2))(x)
    else:
      x = dsample(x)
    return x


class ResBlock(nn.Module):
  """Basic Residual Block."""
  filters: int
  norm_fn: Any
  conv_fn: Any
  dtype: int = jnp.float32
  activation_fn: Any = nn.relu
  use_conv_shortcut: bool = False

  @nn.compact
  def __call__(self, x):
    input_dim = x.shape[-1]
    residual = x
    x = self.norm_fn()(x)
    x = self.activation_fn(x)
    x = self.conv_fn(self.filters, kernel_size=(3, 3), use_bias=False)(x)
    x = self.norm_fn()(x)
    x = self.activation_fn(x)
    x = self.conv_fn(self.filters, kernel_size=(3, 3), use_bias=False)(x)
    if input_dim != self.filters:
      if self.use_conv_shortcut:
        residual = self.conv_fn(
            self.filters, kernel_size=(3, 3), use_bias=False)(
                x)
      else:
        residual = self.conv_fn(
            self.filters, kernel_size=(1, 1), use_bias=False)(
                x)
    return x + residual


class Encoder(nn.Module):
  """Encoder Blocks."""

  config: ml_collections.ConfigDict
  train: bool
  dtype: int = jnp.float32

  def setup(self):
    self.filters = self.config.vqvae.filters
    self.num_res_blocks = self.config.vqvae.num_res_blocks
    self.channel_multipliers = self.config.vqvae.channel_multipliers
    self.embedding_dim = self.config.vqvae.embedding_dim
    self.conv_downsample = self.config.vqvae.conv_downsample
    self.norm_type = self.config.vqvae.norm_type
    if self.config.vqvae.activation_fn == 'relu':
      self.activation_fn = nn.relu
    elif self.config.vqvae.activation_fn == 'swish':
      self.activation_fn = nn.swish
    else:
      raise NotImplementedError

  @nn.compact
  def __call__(self, x):
    conv_fn = nn.Conv
    norm_fn = get_norm_layer(
        train=self.train, dtype=self.dtype, norm_type=self.norm_type)
    block_args = dict(
        norm_fn=norm_fn,
        conv_fn=conv_fn,
        dtype=self.dtype,
        activation_fn=self.activation_fn,
        use_conv_shortcut=False,
    )
    x = conv_fn(self.filters, kernel_size=(3, 3), use_bias=False)(x)
    num_blocks = len(self.channel_multipliers)
    for i in range(num_blocks):
      filters = self.filters * self.channel_multipliers[i]
      for _ in range(self.num_res_blocks):
        x = ResBlock(filters, **block_args)(x)
      if i < num_blocks - 1:
        if self.conv_downsample:
          x = conv_fn(filters, kernel_size=(4, 4), strides=(2, 2))(x)
        else:
          x = dsample(x)
    for _ in range(self.num_res_blocks):
      x = ResBlock(filters, **block_args)(x)
    x = norm_fn()(x)
    x = self.activation_fn(x)
    x = conv_fn(self.embedding_dim, kernel_size=(1, 1))(x)
    return x


class Decoder(nn.Module):
  """Decoder Blocks."""

  config: ml_collections.ConfigDict
  train: bool
  output_dim: int = 3
  dtype: Any = jnp.float32

  def setup(self):
    self.filters = self.config.vqvae.filters
    self.num_res_blocks = self.config.vqvae.num_res_blocks
    self.channel_multipliers = self.config.vqvae.channel_multipliers
    self.norm_type = self.config.vqvae.norm_type
    if self.config.vqvae.activation_fn == 'relu':
      self.activation_fn = nn.relu
    elif self.config.vqvae.activation_fn == 'swish':
      self.activation_fn = nn.swish
    else:
      raise NotImplementedError

  @nn.compact
  def __call__(self, x):
    conv_fn = nn.Conv
    norm_fn = get_norm_layer(
        train=self.train, dtype=self.dtype, norm_type=self.norm_type)
    block_args = dict(
        norm_fn=norm_fn,
        conv_fn=conv_fn,
        dtype=self.dtype,
        activation_fn=self.activation_fn,
        use_conv_shortcut=False,
    )
    num_blocks = len(self.channel_multipliers)
    filters = self.filters * self.channel_multipliers[-1]
    x = conv_fn(filters, kernel_size=(3, 3), use_bias=True)(x)
    for _ in range(self.num_res_blocks):
      x = ResBlock(filters, **block_args)(x)
    for i in reversed(range(num_blocks)):
      filters = self.filters * self.channel_multipliers[i]
      for _ in range(self.num_res_blocks):
        x = ResBlock(filters, **block_args)(x)
      if i > 0:
        x = upsample(x, 2)
        x = conv_fn(filters, kernel_size=(3, 3))(x)
    x = norm_fn()(x)
    x = self.activation_fn(x)
    x = conv_fn(self.output_dim, kernel_size=(3, 3))(x)
    return x

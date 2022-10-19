
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

import ml_collections

import jax
import jax.numpy as jnp
import flax.linen as nn

from nets.enc_dec_arc import Encoder
from nets.enc_dec_arc import Decoder

DEFAULT_PRECISION = jax.lax.Precision.HIGHEST


def get_lax_precision(config):
  lax_precision = config.get('lax_precision', 'default')
  if lax_precision == 'highest':
    return jax.lax.Precision.HIGHEST
  elif lax_precision == 'high':
    return jax.lax.Precision.HIGH
  return jax.lax.Precision.DEFAULT


def squared_euclidean_distance(a: jnp.ndarray,
                               b: jnp.ndarray,
                               b2: jnp.ndarray = None,
                               precision=DEFAULT_PRECISION) -> jnp.ndarray:
  """Computes the pairwise squared Euclidean distance.

  Args:
    a: float32: (n, d): An array of points.
    b: float32: (m, d): An array of points.
    b2: float32: (d, m): b square transpose.
    precision: use HIGHEST precision by default

  Returns:
    d: float32: (n, m): Where d[i, j] is the squared Euclidean distance between
    a[i] and b[j].
  """
  if b2 is None:
    b2 = jnp.sum(b.T**2, axis=0, keepdims=True)
  a2 = jnp.sum(a**2, axis=1, keepdims=True)
  ab = jnp.matmul(a, b.T, precision=precision)
  d = a2 - 2 * ab + b2
  return d


class VectorQuantizer(nn.Module):
  """Basic vector quantizer."""
  config: ml_collections.ConfigDict
  precision: Any
  train: bool = False
  dtype: int = jnp.float32

  def setup(self):
    codebook_size = self.config.vqvae.codebook_size
    embedding_dim = self.config.vqvae.embedding_dim
    self.codebook = self.param(
        'codebook',
        jax.nn.initializers.variance_scaling(
            scale=1.0, mode='fan_in', distribution='uniform'),
        (codebook_size, embedding_dim))

  @nn.compact
  def __call__(self, x, **kwargs):
    codebook_size = self.config.vqvae.codebook_size
    codebook = jnp.asarray(self.codebook, dtype=self.dtype)
    distances = jnp.reshape(
        squared_euclidean_distance(jnp.reshape(x, (-1, x.shape[-1])), codebook),
        x.shape[:-1] + (codebook_size,))
    encoding_indices = jnp.argmin(distances, axis=-1)
    encodings = jax.nn.one_hot(
        encoding_indices, codebook_size, dtype=self.dtype)
    quantized = self.quantize(encodings)
    result_dict = dict()
    result_dict.update({
        'encodings': encodings,
        'encoding_indices': encoding_indices,
        'raw': x,
    })
    return quantized, result_dict

  def quantize(self, z: jnp.ndarray) -> jnp.ndarray:
    codebook = jnp.asarray(self.codebook, dtype=self.dtype)
    return jnp.dot(z, codebook, precision=self.precision)

  def get_codebook(self) -> jnp.ndarray:
    return jnp.asarray(self.codebook, dtype=self.dtype)

  def decode_ids(self, ids: jnp.ndarray) -> jnp.ndarray:
    codebook = self.codebook
    return jnp.take(codebook, ids, axis=0)


def create_quantizer(config: ml_collections.ConfigDict,
                     train: bool,
                     dtype=jnp.float32):
  """Builds an appropriate quantizer from a config."""
  quant_classes = dict(vq=VectorQuantizer)

  def _build_quant():
    """Factory function to build a quantizer based on config."""
    return quant_classes[config.vqvae.quantizer](
        config=config,
        train=train,
        precision=get_lax_precision(config),
        dtype=dtype)

  return _build_quant()


class VQVAE(nn.Module):
  """VQ-VAE model."""
  config: ml_collections.ConfigDict
  train: bool = False
  dtype: int = jnp.float32
  activation_fn: Any = nn.relu

  def setup(self):
    """VQ-VAE setup."""
    self.quantizer = create_quantizer(
        config=self.config, train=self.train, dtype=self.dtype)
    self.precision = get_lax_precision(self.config)

    self.encoder = Encoder(
        config=self.config, train=self.train, dtype=self.dtype)
    self.decoder = Decoder(
        config=self.config, train=self.train, output_dim=3, dtype=self.dtype)

  def encode(self, input_dict):
    image = input_dict['image']
    encoded_feature = self.encoder(image)
    quantized, result_dict = self.quantizer(encoded_feature)
    return quantized, result_dict

  def decode(self, x: jnp.ndarray) -> jnp.ndarray:
    return self.decoder(x)

  def get_codebook_funct(self):
    # This function only works for the naive VQGAN
    return self.quantizer.get_codebook()

  def decode_from_indices(self, inputs):
    if isinstance(inputs, dict):
      ids = inputs['encoding_indices']
    else:
      ids = inputs
    features = self.quantizer.decode_ids(ids)
    reconstructed_image = self.decode(features)
    return reconstructed_image

  def encode_to_indices(self, inputs):
    if isinstance(inputs, dict):
      image = inputs['image']
    else:
      image = inputs
    encoded_feature = self.encoder(image)
    _, result_dict = self.quantizer(encoded_feature)
    ids = result_dict['encoding_indices']
    return ids

  def __call__(self, input_dict):
    quantized, result_dict = self.encode(input_dict)
    outputs = self.decoder(quantized)
    return outputs, result_dict

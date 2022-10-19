
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

from typing import Any, Tuple, Dict, Text
import functools

import jax
import jax.numpy as jnp
import flax.linen as nn

from nets.simplified_bert import BertEmbed
from nets.simplified_bert import BertLayer
from nets.simplified_bert import BertMlmLayer
from nets.simplified_bert import TF_LAYERNORM_EPSILON
from nets.simplified_bert import truncated_normal


class PromptGenerator(nn.Module):
  """Class-conditional Prompt."""
  vocab_size: int
  embedding_size: int = 768
  seq_length: int = 1
  hidden_size: int = 768
  hidden_dropout_prob: float = 0.1
  initializer_range: float = 0.02
  prefix: str = 'prompt'

  def embed_module(self,
                   x: jnp.ndarray,
                   position_embedding: Any = None,
                   deterministic: bool = True,
                   prefix: Text = 'prompt'):

    word_embedder = nn.Embed(
        num_embeddings=self.vocab_size,
        features=self.embedding_size,
        embedding_init=truncated_normal(self.initializer_range),
        name=f'{prefix}_embeddings')
    word_embeddings = word_embedder(x)
    if position_embedding is not None:
      word_embeddings = position_embedding(word_embeddings)
    input_embeddings = nn.LayerNorm(
        epsilon=TF_LAYERNORM_EPSILON, name=f'{prefix}_embeddings_ln')(
            word_embeddings)
    if self.hidden_size > 0:
      input_embeddings = nn.Dense(
          features=self.hidden_size,
          kernel_init=truncated_normal(self.initializer_range),
          name=f'{prefix}_embedding_hidden_mapping')(
              input_embeddings)
    input_embeddings = nn.Dropout(rate=self.hidden_dropout_prob)(
        input_embeddings, deterministic=deterministic)

    return input_embeddings

  def position_embed_module(self,
                            word_embeddings,
                            position_ids: jnp.ndarray = None,
                            position_len: int = 0,
                            prefix: Text = 'word'):
    if position_ids is None:
      return word_embeddings
    position_embeddings = nn.Embed(
        num_embeddings=position_len,
        features=self.embedding_size,
        embedding_init=truncated_normal(self.initializer_range),
        name=f'{prefix}_position_embeddings')(
            position_ids)
    # `position_embeddings`: 1 x position_len x embedding_size,
    # `word_embeddings`: batch_size x 1 x embedding_size.
    return word_embeddings + position_embeddings

  @nn.compact
  def __call__(self,
               x: jnp.ndarray,
               deterministic: bool = True) -> Tuple[jnp.ndarray, jnp.ndarray]:

    position_ids = jnp.arange(self.seq_length)[None, :]
    position_embedding = functools.partial(
        self.position_embed_module,
        position_ids=position_ids,
        position_len=self.seq_length,
        prefix=self.prefix)
    return self.embed_module(
        x=x,
        position_embedding=position_embedding,
        deterministic=deterministic,
        prefix=self.prefix)


class CondBert(nn.Module):
  """BERT as a Flax module.

  In this model, we take cond_embedding as additional tokens for
  transfer learning, while discarding the class-conditional token.
  """
  vocab_size: int
  hidden_size: int = 768
  num_hidden_layers: int = 12
  num_attention_heads: int = 12
  intermediate_size: int = 3072
  hidden_dropout_prob: float = 0.1
  attention_probs_dropout_prob: float = 0.1
  max_position_embeddings: int = 512
  initializer_range: float = 0.02
  pad_token_id: int = -1

  @nn.compact
  def __call__(self,
               input_ids: Tuple[jnp.ndarray, jnp.ndarray],
               deterministic: bool = True) -> Dict[Text, jnp.ndarray]:
    # We assume that all pad tokens should be masked out.
    (input_ids, cond_embeddings) = input_ids
    input_ids = input_ids.astype('int32')
    input_embeddings = BertEmbed(
        embedding_size=self.hidden_size,
        hidden_dropout_prob=self.hidden_dropout_prob,
        vocab_size=self.vocab_size,
        max_position_embeddings=self.max_position_embeddings,
        initializer_fn=truncated_normal(self.initializer_range),
        name='Embed_0')(
            input_ids=input_ids, deterministic=deterministic)

    # Stack BERT layers.
    layer_input = jnp.concatenate(
        (cond_embeddings, input_embeddings[..., 1:, :]), axis=-2)
    input_mask = jnp.ones_like(layer_input, dtype=jnp.int32)[..., 0]
    for i in range(self.num_hidden_layers):
      layer_output = BertLayer(
          intermediate_size=self.intermediate_size,
          hidden_size=self.hidden_size,
          hidden_dropout_prob=self.hidden_dropout_prob,
          num_attention_heads=self.num_attention_heads,
          attention_probs_dropout_prob=self.attention_probs_dropout_prob,
          initializer_fn=truncated_normal(self.initializer_range),
          name=f'TransformerLayer_{i}')(
              layer_input=layer_input,
              input_mask=input_mask,
              deterministic=deterministic)
      layer_input = layer_output

    word_embedding_matrix = self.variables['params']['Embed_0'][
        'word_embeddings']['embedding']
    logits = BertMlmLayer(
        hidden_size=self.hidden_size,
        initializer_fn=truncated_normal(self.initializer_range),
        name='MlmLayer_0')(
            last_layer=layer_output, embeddings=word_embedding_matrix)

    return logits

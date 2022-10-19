
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

from absl import logging
from typing import Any, Callable, Tuple, Optional, Union, Iterable, Dict, Text
import functools

import abc
from clu import metrics
from clu import metric_writers
from clu import parameter_overview
import ml_collections

import numpy as np
import tensorflow as tf
import jax
import jax.numpy as jnp
import flax
import flax.jax_utils as flax_utils
from flax.training import checkpoints as flax_checkpoints
import optax

from nets.vqvae import VQVAE
from nets.simplified_bert import bert_masking
from nets.simplified_bert_prompt import CondBert
from nets.simplified_bert_prompt import PromptGenerator
from trainer.base_trainer import BaseTrainer
from trainer.base_trainer import TrainMetrics
from trainer.base_trainer import TrainState
from maskgit.libml.mask_schedule import schedule as mask_schedule

MASK_TOKEN = -1


def load_model_variables(config, dtype):
  """Loads vqgan and transformer models from the checkpoints."""

  # Load VQGAN checkpoint.
  vqgan_ckpt_path = config.vqgan_ckpt_path
  state_dict = flax_checkpoints.restore_checkpoint(vqgan_ckpt_path, None)
  vqvae_variables = {'params': flax.core.FrozenDict(state_dict['params'])}
  vqvae_model = VQVAE(
      config=config, dtype=config.get('dtype'), train=False)

  # Load MaskGIT checkpoint.
  transformer_ckpt_path = config.transformer_ckpt_path
  with tf.io.gfile.GFile(transformer_ckpt_path, 'rb') as f:
    transformer_state_dict = flax.serialization.msgpack_restore(f.read())
  transformer_variables = {
      'params': flax.core.FrozenDict(transformer_state_dict['params'])
  }
  # Class-conditional ImageNet pretrained MaskGIT model.
  transformer_model = CondBert(
      vocab_size=config.vqvae.codebook_size + 1000 + 1,
      hidden_size=config.transformer.num_embeds,
      num_hidden_layers=config.transformer.num_layers,
      num_attention_heads=config.transformer.num_heads,
      intermediate_size=config.transformer.intermediate_size,
      hidden_dropout_prob=config.transformer.dropout_rate,
      attention_probs_dropout_prob=config.transformer.dropout_rate,
      max_position_embeddings=config.transformer.latent_size**2 + 1,
      pad_token_id=config.transformer.pad_token_id)
  return vqvae_model, vqvae_variables, transformer_model, transformer_variables


def encode_images_to_tokens(batch, mask_ratio, vqvae_model, vqvae_variables,
                            config):
  """Encodes images into latent indices using the pretrained VQGAN model."""
  _, g_dict = vqvae_model.apply(
      vqvae_variables, batch, method=vqvae_model.encode, mutable=False)
  z_indices = g_dict['encoding_indices']

  latent_indices = jnp.reshape(z_indices, (z_indices.shape[0], -1))

  # class_idx is in [codebook_size, codebook_size + 1000)
  class_idx = batch['label'] + config.vqvae.codebook_size
  class_idx = class_idx[:, None].astype(latent_indices.dtype)

  # Randomly mask out.
  masked_dict = bert_masking(
      latent_indices,
      jnp.arange(0, config.vqvae.codebook_size),
      MASK_TOKEN,
      mask_rate=mask_ratio,
      mask_token_proportion=1.,
      random_token_proportion=0.)

  target_latent_indices = jnp.concatenate((class_idx, masked_dict['targets']),
                                          axis=-1)
  # Weights are binary: [batch, seq_len], where 0 represents known, 1 otherwise.
  weights = jnp.concatenate((jnp.zeros_like(class_idx), masked_dict['weights']),
                            axis=-1)

  masked_latent_indices = jnp.concatenate(
      (class_idx, masked_dict['masked_inputs']), axis=-1)

  token_dict = dict(
      input_indices=masked_latent_indices,
      output_indices=target_latent_indices,
      class_idx=class_idx,
      weights=weights)
  batch.update(token_dict)
  return batch


def apply_label_smoothing(one_hot_targets: jnp.ndarray,
                          label_smoothing: float) -> jnp.ndarray:
  """Apply label smoothing to the one-hot targets.

  Applies label smoothing such that the on-values are transformed from 1.0 to
  `1.0 - label_smoothing + label_smoothing / num_classes`, and the off-values
  are transformed from 0.0 to `label_smoothing / num_classes`.
  https://arxiv.org/abs/1512.00567

  Note that another way of performing label smoothing (which we don't use here)
  is to take `label_smoothing` mass from the on-values and distribute it to the
  off-values; in other words, transform the on-values to `1.0 - label_smoothing`
  and the  off-values to `label_smoothing / (num_classes - 1)`.
  http://jmlr.org/papers/v20/18-789.html


  Args:
    one_hot_targets: One-hot targets for an example, a [batch, ..., num_classes]
      float array.
    label_smoothing: A scalar in [0, 1] used to smooth the labels.

  Returns:
    A float array of the same shape as `one_hot_targets` with smoothed label
    values.
  """
  on_value = 1.0 - label_smoothing
  num_classes = one_hot_targets.shape[-1]
  off_value = label_smoothing / num_classes
  one_hot_targets = one_hot_targets * on_value + off_value
  return one_hot_targets


def weighted_sequence_cross_entropy_loss(
    *,
    labels: jnp.ndarray,
    logits: jnp.ndarray,
    weights: jnp.ndarray,
    label_smoothing: Optional[float] = 0.0):
  """Computes the mean cross-entropy for the sequence predictions.

  Args:
    labels: 2D int array of shape (B, T) where each value is in [0, C-1].
    logits: 3D array of shape (B, T, C) where C is number of classes.
    weights: 2D float array (B, T).
    label_smoothing: float.

  Returns:
    float loss.
  """
  vocab_size = logits.shape[-1]
  one_hot_targets = jax.nn.one_hot(labels, vocab_size)
  soft_targets = apply_label_smoothing(one_hot_targets, label_smoothing)
  loss = -jnp.sum(soft_targets * jax.nn.log_softmax(logits), axis=-1)
  loss = jnp.sum(
      loss * weights, axis=-1) / jnp.clip(jnp.sum(weights, axis=-1), 1e-8)
  return jnp.mean(loss)


class PromptTrainer(BaseTrainer):
  """Prompt tuning of MaskGIT."""

  def __init__(self, config: ml_collections.ConfigDict, workdir: str):
    super().__init__(config, workdir)
    if True:
      (vqvae_model, vqvae_variables, transformer_model,
       transformer_variables) = load_model_variables(self.config, self.dtype)
      self.transformer_model = transformer_model
      self.transformer_variables = transformer_variables

      self.tokenizer = functools.partial(
          encode_images_to_tokens,
          vqvae_model=vqvae_model,
          vqvae_variables=vqvae_variables,
          config=self.config,
      )

  def generate_mask_ratio(self, rng):
    ratio = jax.random.uniform(rng, [1])[0]
    return mask_schedule(
        ratio,
        self.config.transformer.latent_size**2,
        method=self.config.mask_scheduling_method)

  def create_optimizer(self, learning_rate=None):
    """Creates a simple optimizer."""
    return optax.adamw(
        learning_rate=learning_rate or self.config.optimizer.lr,
        b1=self.config.optimizer.beta1,
        b2=self.config.optimizer.beta2,
        weight_decay=0.0)

  def create_learning_rate_schedule(self):
    return optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=self.config.optimizer.lr,
        warmup_steps=0,
        decay_steps=max(self.config.num_train_steps, 1))

  def create_train_state(
      self,
      rng: np.ndarray,
      inputs: Dict[str, np.ndarray],
  ):
    """Creates a model and its state."""
    model = PromptGenerator(
        vocab_size=self.config.num_class,
        embedding_size=self.config.prompt.embedding_size,
        seq_length=self.config.prompt.seq_length,
        hidden_size=self.config.get('prompt.hidden_size', 0),
        hidden_dropout_prob=self.config.get(
            'prompt.dropout_rate', self.config.transformer.dropout_rate),
        prefix='prompt')

    init_rng, rng = jax.random.split(rng, 2)
    model_state = model.init(init_rng, inputs['label'][..., None].astype(
        jnp.int16)).unfreeze()
    logging.info('logging model parameters')
    parameter_overview.log_parameter_overview(model_state['params'])
    self.lr_fn = self.create_learning_rate_schedule()
    self.optimizer = self.create_optimizer(learning_rate=self.lr_fn)
    optimizer_state = self.optimizer.init(model_state)
    model_dict = dict(model=model)
    train_state = TrainState(
        step=0, optimizer_state=optimizer_state, model_state=model_state)
    return model_dict, train_state

  def train_step(
      self,
      rng: np.ndarray,
      state: Any,
      batch: Dict[str, jnp.ndarray],
      model_dict: Dict[str, Any],
  ) -> Tuple[Any, metrics.Collection]:
    """Performs a single training step.

    Args:
      rng: The random seed,
      state: State of the model (optimizer and state).
      batch: Training inputs for this step.
      model_dict: The model used in training.

    Returns:
      The new model state and dictionary with metrics
    """
    logging.info('train_step(batch=%s)', batch)
    mask_rng, dropout_rng, rng = jax.random.split(rng, 3)
    mask_ratio = self.generate_mask_ratio(mask_rng)
    batch = self.tokenizer(batch, mask_ratio)

    def loss_fn(variables):
      cond_embeddings, new_variables = model_dict['model'].apply(
          variables,
          batch['label'][..., None],
          deterministic=False,
          mutable=True,
          rngs={'dropout': dropout_rng})
      logits = self.transformer_model.apply(
          self.transformer_variables, (batch['input_indices'], cond_embeddings),
          deterministic=True)
      logits = logits[:, -(self.config.transformer.latent_size**2 +
                           1):, :self.config.vqvae.codebook_size]
      loss = weighted_sequence_cross_entropy_loss(
          labels=batch['output_indices'],
          logits=logits,
          weights=batch['weights'],
          label_smoothing=self.config.label_smoothing)
      loss = jnp.mean(loss)
      return loss, new_variables

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    lr = self.lr_fn(state.step)
    (loss, new_model_state), grad = grad_fn(state.model_state)
    new_model_state = new_model_state.unfreeze()
    grad = jax.lax.pmean(grad, 'batch')

    updates, new_opt_state = self.optimizer.update(grad, state.optimizer_state,
                                                   new_model_state)
    new_model_state = optax.apply_updates(new_model_state, updates)
    new_state = state.replace(
        step=state.step + 1,
        optimizer_state=new_opt_state,
        model_state=new_model_state,
    )
    metrics_update = TrainMetrics.gather_from_model_output(loss=loss, lr=lr)
    return new_state, metrics_update

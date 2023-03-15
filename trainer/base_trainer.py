
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

import os
from absl import logging
from typing import Any, Callable, Tuple, Optional, Union, Iterable, Dict, Text
import functools

import abc
from clu import metrics
from clu import metric_writers
from clu import periodic_actions
import ml_collections

import numpy as np
import tensorflow as tf
import jax
import jax.numpy as jnp
import flax
import flax.jax_utils as flax_utils
from flax.training import checkpoints as flax_checkpoints

from libml.input_pipeline import create_datasets


@flax.struct.dataclass
class TrainState:
  """Data structure for checkpoint the model."""
  step: int
  optimizer_state: Any
  model_state: Optional[Any]


@flax.struct.dataclass
class TrainMetrics(metrics.Collection):
  """Data structure for metrics."""
  loss: metrics.Average.from_output('loss')
  lr: metrics.Average.from_output('lr')


def save_checkpoint(state, workdir, keep=100, keep_every_n_steps=None):
  # get train state from the first replica
  state = jax.device_get(jax.tree_map(lambda x: x[0], state))
  step = int(state.step)
  flax_checkpoints.save_checkpoint_multiprocess(
      workdir,
      state,
      step,
      keep=keep,
      overwrite=True,
      keep_every_n_steps=keep_every_n_steps)


class BaseTrainer(abc.ABC):
  """Base trainer model."""

  def __init__(self, config: ml_collections.ConfigDict, workdir: str):
    self.config = config
    self.workdir = workdir
    self.rng = jax.random.PRNGKey(config.seed)
    self.dtype, self.data_dtype = self.get_dtype()

  def get_dtype(self):
    if self.config.dtype == 'bfloat16':
      return jnp.bfloat16, tf.bfloat16
    else:
      return jnp.float32, tf.float32

  def next_train_data(self, train_iter):
    return jax.tree_map(np.asarray, next(train_iter))

  @abc.abstractmethod
  def create_train_state(
      self,
      rng: np.ndarray,
      inputs: Dict[str, np.ndarray],
  ):
    pass

  def merge_model_state(self, state):
    if jax.tree_leaves(state.model_state):
      cross_replica_mean = jax.pmap(lambda x: jax.lax.pmean(x, 'x'), 'x')
      return state.replace(model_state=cross_replica_mean(state.model_state))
    else:
      return state

  @abc.abstractmethod
  def train_step(self, rng, state, batch, model_dict):
    pass

  def train(self):
    """Training loop."""

    tf.io.gfile.makedirs(self.workdir)
    checkpoint_dir = os.path.join(self.workdir, 'checkpoints')

    rng, data_rng = jax.random.split(self.rng)

    # Data loader.
    data_rng = jax.random.fold_in(data_rng, jax.process_index())
    train_ds, _, ds_metadata = create_datasets(self.config, data_rng)
    steps_per_epoch = max(
        ds_metadata['num_train_examples'] // self.config.batch_size, 1)
    with self.config.unlocked():
      self.config.num_class = ds_metadata['num_classes']
      self.config.num_train_steps = (
          self.config.num_train_epochs * steps_per_epoch)
      self.config.checkpoint_every_steps = self.config.get(
          'checkpoint_every_epochs', 1) * steps_per_epoch
      self.config.log_every_steps = self.config.get(
          'log_every_epochs', 1) * steps_per_epoch
    logging.info('total_steps=%d', self.config.num_train_steps)

    rng, model_rng = jax.random.split(rng)
    train_iter = iter(train_ds)  # pytype: disable=wrong-arg-types
    init_batch = jax.tree_map(np.asarray, next(train_iter))
    model_dict, state = self.create_train_state(
        model_rng, jax.tree_map(lambda x: x[0], init_batch))
    del init_batch
    state = flax_checkpoints.restore_checkpoint(checkpoint_dir, state)

    # If this is the first step, or the model was warm-started, init checkpoint.
    initial_step = int(state.step) + 1  # pytype: disable=attribute-error
    state = flax_utils.replicate(state)
    writer = metric_writers.create_default_writer(
        self.workdir, just_logging=jax.process_index() > 0)
    if initial_step == 1:
      save_checkpoint(state, checkpoint_dir)
      writer.write_hparams(dict(self.config))

    logging.info('Starting training loop at step %d.', initial_step)

    report_progress = periodic_actions.ReportProgress(
        num_train_steps=self.config.num_train_steps, writer=writer)
    hooks = [report_progress] if jax.process_index() == 0 else []

    p_train_step = jax.pmap(
        functools.partial(
            self.train_step,
            model_dict=model_dict,
        ),
        axis_name='batch')

    train_metrics = None
    rng = jax.random.fold_in(rng, jax.process_index())
    rng, train_rng, sample_rng = jax.random.split(rng, 3)  # pylint: disable=unused-variable

    with metric_writers.ensure_flushes(writer):
      for step in range(initial_step, self.config.num_train_steps + 1):
        # `step` is a Python integer. `state.step` is JAX integer on the GPU/TPU
        # devices.
        is_last_step = step == self.config.num_train_steps
        batch = self.next_train_data(train_iter)
        step_rng = jax.random.fold_in(train_rng, step)
        step_rngs = jax.random.split(step_rng, jax.local_device_count())
        state, metrics_update = p_train_step(step_rngs, state, batch)
        metric_update = flax.jax_utils.unreplicate(metrics_update)
        train_metrics = (
            metric_update
            if train_metrics is None else train_metrics.merge(metric_update))

        # Quick indication that training is happening.
        logging.log_first_n(logging.INFO, 'Finished training step %d.', 5, step)
        for h in hooks:
          h(step)

        if step % self.config.checkpoint_every_steps == 0:  # or is_last_step:
          with report_progress.timed('checkpoint'):
            state = self.merge_model_state(state)
            save_checkpoint(state, checkpoint_dir)

        if step % self.config.log_every_steps == 0 or is_last_step:
          writer.write_scalars(step, train_metrics.compute())
          train_metrics = None

      logging.info('Finishing training at step %d', self.config.num_train_steps)

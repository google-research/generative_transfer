
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
from typing import Tuple, Dict

from clu import deterministic_data
import ml_collections

import tensorflow as tf
import jax

from task_adaptation import data_loader as task_adapt_loader

from dataset.vtab import VtabTfds
from libml.preprocess import preprocess_data


def create_datasets(
    config: ml_collections.ConfigDict,
    data_rng,
) -> Tuple[tf.data.Dataset, tf.data.Dataset, Dict[str, int]]:
  """Create datasets for training and evaluation.

  For the same data_rng and config this will return the same datasets. The
  datasets only contain stateless operations. See go/deterministic-training to
  learn how this helps with reproducible training.

  Args:
    config: Configuration to use.
    data_rng: PRNGKey for seeding operations in the training dataset.

  Returns:
    The training dataset, the evaluation dataset and num_training samples.
  """
  if config.batch_size % jax.device_count() != 0:
    raise ValueError(f'Batch size ({config.batch_size}) must be divisible by '
                     f'the number of devices ({jax.device_count()}).')
  per_device_batch_size_train = config.batch_size // jax.device_count()
  logging.info('per_device_batch_size_train %d !!', per_device_batch_size_train)

  dataset_name = config.dataset
  # Load VTAB dataset.
  dataset_builder = task_adapt_loader.get_dataset_instance({
      'dataset': f'data.{dataset_name}',
      'data_dir': None
  })
  dataset_builder = VtabTfds(dataset_builder)
  num_classes = dataset_builder.info.num_classes

  train_str = 'train'
  eval_str = 'all'

  num_train_examples = dataset_builder.info.splits[train_str].num_examples
  num_validation_examples = dataset_builder.info.splits[
      'trainval'].num_examples + dataset_builder.info.splits['test'].num_examples

  train_split = train_str
  eval_split = eval_str

  preprocess_fn_train = preprocess_data(
      config, train_flag=True, num_classes=num_classes)
  preprocess_fn_eval = preprocess_data(
      config, train_flag=False, num_classes=num_classes)

  decoders = None

  train_data_rng, eval_data_rng = jax.random.split(data_rng, 2)
  
  train_ds = deterministic_data.create_dataset(
      dataset_builder,
      split=train_split,
      rng=train_data_rng,
      preprocess_fn=preprocess_fn_train,
      cache=False,
      decoders=None,
      shuffle_buffer_size=config.shuffle_buffer_size,
      batch_dims=[jax.local_device_count(), per_device_batch_size_train],
      num_epochs=None,
      shuffle=config.train_shuffle,
  )

  eval_num_batches = None
  eval_batch_size_per_replica = config.eval_batch_size // jax.device_count()
  eval_ds = deterministic_data.create_dataset(
      dataset_builder,
      split=eval_split,
      rng=eval_data_rng,
      preprocess_fn=preprocess_fn_eval,
      cache=jax.process_count() > 1,
      batch_dims=[jax.local_device_count(), eval_batch_size_per_replica],
      num_epochs=None,
      shuffle_buffer_size=config.shuffle_buffer_size,
      shuffle=True,
      pad_up_to_batches=eval_num_batches,
  )

  options = tf.data.Options()
  options.experimental_external_state_policy = (
      tf.data.experimental.ExternalStatePolicy.WARN)
  train_ds = train_ds.with_options(options)
  eval_ds = eval_ds.with_options(options)
  metadata_info = {
      'num_train_examples': num_train_examples,
      'num_validation_examples': num_validation_examples,
      'num_classes': num_classes
  }
  return train_ds, eval_ds, metadata_info

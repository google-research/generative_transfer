
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

from clu import deterministic_data
from clu.deterministic_data import DatasetBuilder

import ml_collections


NATURAL = [
    'caltech101', 'cifar(num_classes=100)', 'sun397', 'svhn',
    'oxford_flowers102', 'oxford_iiit_pet', 'dtd'
]
SPECIALIZED = [
    'eurosat', 'resisc45', 'patch_camelyon',
    'diabetic_retinopathy(config="btgraham-300")'
]
STRUCTURED = [
    'kitti(task="closest_vehicle_distance")',
    'smallnorb(predicted_attribute="label_azimuth")',
    'smallnorb(predicted_attribute="label_elevation")',
    'dsprites(predicted_attribute="label_x_position",num_classes=16)',
    'dsprites(predicted_attribute="label_orientation",num_classes=16)',
    'clevr(task="closest_object_distance")',
    'clevr(task="count_all")',
    'dmlab',
]
DATASETS = NATURAL + SPECIALIZED + STRUCTURED


class VtabTfds(DatasetBuilder):
  """VTAB data object."""

  def __init__(self, dataset):
    self.dataset = dataset
    self.info = ml_collections.ConfigDict({
        'splits': {
            split: {
                'num_examples': dataset.get_num_samples(split)
            } for split in dataset.splits
        },
        'num_classes': dataset.get_num_classes()
    })

  def as_dataset(self, split, shuffle_files, read_config, decoders):
    """Returns tf.data object.

    Args:
      split: str, 'train' or 'val'.
      shuffle_files: bool, whether to shuffle files.
      read_config: tf.data.ReadConfig.
      decoders: dict, data decoders.

    Returns:
      tf.data object.
    """
    # pylint: disable=protected-access
    for_eval = 'train' not in split
    train_examples = self.dataset.get_num_samples(
        split) if 'train' in split else None

    if split == 'all':
      data1 = self.dataset._dataset_builder.as_dataset(
          split=self.dataset._tfds_splits['trainval'],
          shuffle_files=shuffle_files,
          read_config=read_config,
          decoders=decoders)
      data2 = self.dataset._dataset_builder.as_dataset(
          split=self.dataset._tfds_splits['test'],
          shuffle_files=shuffle_files,
          read_config=read_config,
          decoders=decoders)
      data = data1.concatenate(data2)
    else:
      data = self.dataset._dataset_builder.as_dataset(
          split=self.dataset._tfds_splits[split],
          shuffle_files=shuffle_files,
          read_config=read_config,
          decoders=decoders)

    if not for_eval:
      # Deterministic for same dataset version.
      data = data.take(train_examples)
      num_samples = train_examples
    elif split == 'all':
      num_samples = self.dataset.get_num_samples(
          'trainval') + self.dataset.get_num_samples('test')
    else:
      num_samples = self.dataset.get_num_samples(split)

    data = self.dataset._cache_data_if_possible(
        data, split_name=split, num_samples=num_samples, for_eval=for_eval)

    data = data.map(self.dataset._base_preprocess_fn,
                    self.dataset._num_preprocessing_threads)

    return data.prefetch(1)

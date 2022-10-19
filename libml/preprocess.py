
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

from typing import Callable, Dict, Sequence, Tuple
import functools

import tensorflow as tf

FeatureDict = Dict[str, tf.Tensor]
MapFn = Callable[[FeatureDict], FeatureDict]


def convert_to_rgb(image: tf.Tensor) -> tf.Tensor:
  """Converts images with 1 or 4 channels to have 3 channels."""
  
  shape = tf.shape(image)
  image = tf.cond(
      tf.shape(tf.shape(image))[0] > 3, lambda: image[0], lambda: image)
  # For grayscale images, simply convert to RGB.
  image = tf.cond(shape[-1] < 3,
                  lambda: tf.image.grayscale_to_rgb(image[..., :1]),
                  lambda: image)
  # For RGBA images, drop the alpha channel.
  image = tf.cond(tf.equal(shape[-1], 4), lambda: image[..., :3], lambda: image)
  image = image[..., :3]
  image.set_shape([None, None, 3])
  return image


def central_crop(image: tf.Tensor, height: int, width: int) -> tf.Tensor:
  """Makes central crop of a given size."""
  top = (tf.shape(image)[0] - height) // 2
  left = (tf.shape(image)[1] - width) // 2
  image = tf.image.crop_to_bounding_box(image, top, left, height, width)
  return image


def random_crop_and_resize_without_stretch(
    image: tf.Tensor,
    rng,
    resize_height: int,
    resize_width: int,
    upsample_scale: float = 1.0) -> Tuple[tf.Tensor, Sequence[int]]:
  """Resizes without changing aspect ratio and random crop the images."""
  h, w = tf.shape(image)[0], tf.shape(image)[1]

  # Figure out the necessary h/w.
  ratio_h = tf.cast(resize_height, tf.float32) * upsample_scale / tf.cast(
      h, tf.float32)
  ratio_w = tf.cast(resize_width, tf.float32) * upsample_scale / tf.cast(
      w, tf.float32)
  ratio = tf.maximum(ratio_h, ratio_w)

  h = tf.cast(tf.round(tf.cast(h, tf.float32) * ratio), tf.int32)
  w = tf.cast(tf.round(tf.cast(w, tf.float32) * ratio), tf.int32)

  image = tf.image.resize(image, [h, w], antialias=True)

  image = tf.image.stateless_random_crop(
      image, (resize_height, resize_width, 3), seed=rng)
  image = tf.cast(image, tf.float32)
  image = tf.image.resize(image, (resize_height, resize_width), antialias=True)
  return image


def train_preprocess(features: FeatureDict, image_height: int, image_width: int,
                     dtype: int, num_classes: int,
                     map_fn: MapFn) -> FeatureDict:
  """Processes a single example for training."""
  image = features['image']
  # This PRNGKey is unique to this example. We can use it with the stateless
  # random ops in TF.
  rng = features.pop('rng')
  rng, rng_crop, rng_flip = tf.unstack(
      tf.random.experimental.stateless_split(rng, 3))
  # Normalize into [0, 1]
  image = tf.cast(image, tf.float32) / 255.0
  image = convert_to_rgb(image)
  image = random_crop_and_resize_without_stretch(
      image, rng_crop, resize_height=image_height, resize_width=image_width)
  image = tf.image.stateless_random_flip_left_right(image, rng_flip)
  image = tf.cast(image, dtype)
  if 'label' in features:
    one_hot_label = tf.one_hot(features['label'], num_classes)
    one_hot_label = tf.cast(one_hot_label, dtype)
    label = tf.cast(features['label'], tf.int32)
    image_dict = {
        'image': image,
        'label': label,
    }
  else:
    image_dict = {'image': image}
  return map_fn(image_dict)


def eval_preprocess(features: FeatureDict, image_height: int, image_width: int,
                    dtype: int, num_classes: int, map_fn: MapFn) -> FeatureDict:
  """Process a single example for evaluation."""
  image = features['image']
  # Normalize into [0, 1]
  image = tf.cast(image, tf.float32) / 255.0
  image = convert_to_rgb(image)
  min_size = tf.minimum(tf.shape(image)[0], tf.shape(image)[1])
  image = central_crop(image, min_size, min_size)
  image = tf.image.resize(image, [image_height, image_width], antialias=True)
  image = tf.cast(image, dtype)
  if 'label' in features:
    one_hot_label = tf.one_hot(features['label'], num_classes)
    one_hot_label = tf.cast(one_hot_label, dtype)
    label = tf.cast(features['label'], tf.int32)
    image_dict = {
        'image': image,
        'label': label,
    }
  else:
    image_dict = {'image': image}
  return map_fn(image_dict)


def preprocess_data(config, train_flag, num_classes=None):
  """Preprocessing data."""
  if config.get('image_height') and config.get('image_width'):
    image_height = config.image_height
    image_width = config.image_width
  elif config.get('image_size'):
    image_height = image_width = config.image_size
  else:
    raise NotImplementedError

  if config.dtype == 'bfloat16':
    dtype = tf.bfloat16
  else:
    dtype = tf.float32

  map_fn = lambda x: x

  if train_flag:
    return functools.partial(
        train_preprocess,
        image_height=image_height,
        image_width=image_width,
        dtype=dtype,
        num_classes=num_classes,
        map_fn=map_fn)
  else:
    return functools.partial(
        eval_preprocess,
        image_height=image_height,
        image_width=image_width,
        dtype=dtype,
        num_classes=num_classes,
        map_fn=map_fn)

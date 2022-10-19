
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

import math
import matplotlib.pyplot as plt


def visualize_image(image, title='', figsize=(10, 10), format='png'):
  plt.figure(figsize=figsize)
  plt.imshow(image)
  plt.axis('off')
  if len(title) > 0:
    plt.title(title)
  plt.savefig(
      f'image.{format}', format=format, bbox_inches='tight', pad_inches=0)


def make_grid(samples, show_num=64):
  """Tile images to an image grid."""
  batch_size, height, width, c = samples.shape
  if batch_size < show_num:
    logging.info('show_num is cut by the small batch size to %d', batch_size)
    show_num = batch_size
  h_num = int(math.sqrt(show_num))
  w_num = int(show_num / h_num)
  grid_num = h_num * w_num

  samples = samples[0:grid_num]
  samples = samples.reshape(h_num, w_num, height, width, c)
  samples = samples.swapaxes(1, 2)
  samples = samples.reshape(height * h_num, width * w_num, c)
  return samples

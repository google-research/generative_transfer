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
from typing import Any
import functools
import ml_collections

import numpy as np
import jax
import jax.numpy as jnp
from flax.training import checkpoints as flax_checkpoints

from nets.simplified_bert_prompt import PromptGenerator
from libml import parallel_decode
from trainer.prompt_trainer import PromptTrainer
from trainer.prompt_trainer import load_model_variables
from utils.vis import visualize_image

TRANSFORMER_CKPT_PATH = 'checkpoints/maskgit_imagenet256_checkpoint'
TOKENIZER_CKPT_PATH = 'checkpoints/tokenizer_imagenet256_checkpoint'


def get_default_config():
  """Gets model configuration."""
  
  vqvae_config = ml_collections.ConfigDict()
  vqvae_config.codebook_size = 1024
  vqvae_config.embedding_dim = 256
  vqvae_config.quantizer = 'vq'
  vqvae_config.filters = 128
  vqvae_config.num_res_blocks = 2
  vqvae_config.channel_multipliers = [1, 1, 2, 2, 4]
  vqvae_config.embedding_dim
  vqvae_config.conv_downsample = False
  vqvae_config.norm_type = 'GN'
  vqvae_config.activation_fn = 'swish'

  bert_config = ml_collections.ConfigDict()
  bert_config.num_embeds = 768
  bert_config.num_heads = 16
  bert_config.num_layers = 24
  bert_config.intermediate_size = 768 * 4
  bert_config.dropout_rate = 0.1
  bert_config.latent_size = 16
  bert_config.pad_token_id = -1

  config = ml_collections.ConfigDict()
  config.vqgan_ckpt_path = TOKENIZER_CKPT_PATH
  config.transformer_ckpt_path = TRANSFORMER_CKPT_PATH
  config.vqvae = vqvae_config
  config.transformer = bert_config

  # prompt config
  config.prompt = ml_collections.ConfigDict()
  config.prompt.embedding_size = config.transformer.num_embeds
  config.prompt.hidden_size = 0
  config.prompt.seq_length = 32
  config.prompt.dropout_rate = 0.1

  # optimizer
  config.optimizer = ml_collections.ConfigDict()
  config.optimizer.lr = 0.001
  config.optimizer.beta1 = 0.9
  config.optimizer.beta2 = 0.96
  config.optimizer.warmup_steps = 0
  config.optimizer.weight_decay = 0
  config.label_smoothing = 0.1
  config.mask_scheduling_method = 'cosine'
  config.seed = 42
  config.dtype = 'float32'

  config.batch_size = 32
  config.eval_batch_size = 32
  config.num_train_epochs = 100
  config.checkpoint_every_epochs = 10
  config.log_every_epochs = 1

  # dataset
  config.dataset = 'caltech101'
  config.image_size = 256
  config.shuffle_buffer_size = 1000
  config.train_shuffle = True

  config.lock()

  return config


def load_prompt_from_checkpoint(checkpoint_dir, config):
  """Loads prompt generator from the checkpoint."""

  prompt_model = PromptGenerator(
      vocab_size=config.num_class,
      embedding_size=config.prompt.embedding_size,
      hidden_size=config.get("prompt.hidden_size",
                             config.transformer.num_embeds),
      hidden_dropout_prob=config.get("prompt.dropout_rate",
                                     config.transformer.dropout_rate),
      seq_length=config.prompt.seq_length,
      prefix="prompt")
  prompt_state = flax_checkpoints.restore_checkpoint(checkpoint_dir, None)
  return prompt_model, prompt_state


def prompt_decode(code_input,
                  cond_input,
                  rng,
                  num_iter=12,
                  choice_temperature=4.5,
                  config=Any,
                  transformer_model=Any,
                  transformer_variables=Any,
                  prompt_model=Any,
                  prompt_variables=Any):
  """Decodes (synthesis) with prompt."""

  def tokens_to_logits(seq):

    logits = transformer_model.apply(
        transformer_variables, (seq, cond_embeddings), deterministic=True)
    logits = logits[:, -(config.transformer.latent_size**2 +
                         1):, :config.vqvae.codebook_size]
    return logits

  cond_embeddings = prompt_model.apply(
      prompt_variables, cond_input, mutable=False, deterministic=True)

  # output size is [batch_size, num_iter, seq_lenth]
  output_indices = parallel_decode.decode(
      code_input,
      rng,
      tokens_to_logits,
      num_iter=num_iter,
      choice_temperature=choice_temperature)
  return output_indices


def detokenizer(indices, target_shape, vqvae_model, vqvae_variables):
  """Decodes latent indices to images using VQGAN model."""

  indices = jnp.reshape(indices, target_shape)
  return vqvae_model.apply(
      vqvae_variables,
      indices,
      method=vqvae_model.decode_from_indices,
      mutable=False)


class Sampler():
  """Synthesize images."""

  def __init__(self, config, checkpoint_dir):
    self.config = config
    self.checkpoint_dir = checkpoint_dir

    # Load pretrained models.
    (vqvae_model, vqvae_variables, transformer_model,
     transformer_variables) = load_model_variables(self.config,
                                                   self.config.dtype)

    # Load prompt.
    (prompt_model,
     prompt_state) = load_prompt_from_checkpoint(self.checkpoint_dir,
                                                 self.config)

    self.sample_tokens_pmap = jax.pmap(
        functools.partial(
            prompt_decode,
            config=self.config,
            transformer_model=transformer_model,
            transformer_variables=transformer_variables,
            prompt_model=prompt_model,
            prompt_variables=prompt_state['model_state']),
        in_axes=0,
        donate_argnums=(1,),
        static_broadcasted_argnums=(3, 4))
    self.detokenizer_pmap = jax.pmap(
        functools.partial(
            detokenizer,
            target_shape=[
                -1, config.transformer.latent_size,
                config.transformer.latent_size
            ],
            vqvae_model=vqvae_model,
            vqvae_variables=vqvae_variables),
        in_axes=0,
        donate_argnums=(1,))

  def get_dummy_input(self, device_batch_size=4):
    """Gets dummy input for generation."""

    num_devices = jax.device_count()
    batch = {
        'label':
            jnp.tile(
                jnp.arange(num_devices)[..., None], (1, device_batch_size)),
        'code':
            -1 * jnp.ones([
                num_devices, device_batch_size,
                self.config.transformer.latent_size**2 + 1
            ])
    }
    return batch['code'], batch['label'][..., None]

  def sample(self, rng, num_iter=12, temperature=4.5, device_batch_size=4):
    """Samples image."""
    num_devices = jax.device_count()
    code_input, cond_input = self.get_dummy_input(device_batch_size)
    output_indices = self.sample_tokens_pmap(code_input, cond_input,
                                             jax.random.split(rng, num_devices),
                                             num_iter, temperature)
    outputs = jnp.array(output_indices, dtype=jnp.int32)
    outputs = jnp.array(
        output_indices[:, :, -1, -self.config.transformer.latent_size**2:],
        dtype=jnp.int32)
    return self.detokenizer_pmap(
        outputs)  # num_devices x device_batch_size x imsize

  def visualize(self, gen_images):
    """Visualizes generated image."""

    gen_images = np.reshape(
        np.transpose(gen_images, [0, 2, 1, 3, 4]), [
            gen_images.shape[0] * gen_images.shape[2],
            gen_images.shape[1] * gen_images.shape[3], gen_images.shape[4]
        ])
    visualize_image(gen_images, figsize=(30, 30))

  def sample_and_visualize(self,
                           rng,
                           num_iter=12,
                           temperature=4.5,
                           device_batch_size=4):
    gen_images = self.sample(rng, num_iter, temperature, device_batch_size)
    self.visualize(gen_images)
    return gen_images
  

def main():
  """Main."""
  workdir = './results/1'
  config = get_default_config()
  with config.unlocked():
    # Change default values in config
    config.seed = 43

  # Run prompt tuning.
  trainer = PromptTrainer(config, workdir)
  trainer.train()

  # Sample.
  checkpoint_dir = os.path.join(workdir, 'checkpoints')
  sampler = Sampler(trainer.config, checkpoint_dir)
  gen_images = sampler.sample(rng=jax.random.PRNGKey(0))
  
  # Visualize.
  gen_images = np.reshape(
      np.transpose(gen_images, [0, 2, 1, 3, 4]), [
          gen_images.shape[0] * gen_images.shape[2],
          gen_images.shape[1] * gen_images.shape[3], gen_images.shape[4]
      ])
  visualize_image(gen_images, figsize=(30, 30))

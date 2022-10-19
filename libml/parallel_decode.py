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

"""Fast decoding routines for non-autoregressive generation."""

import jax
import jax.numpy as jnp

from maskgit.libml import mask_schedule
from maskgit.libml import parallel_decode

_CONFIDENCE_OF_KNOWN_TOKENS = jnp.inf


def mask_by_random_topk(rng, mask_len, probs, temperature=1.0):
  """Modifies from jax.random.choice without replacement.
  JAX's original implementation is as below:
    g = -gumbel(key, (n_inputs,)) - jnp.log(p)
    ind = jnp.argsort(g)[:n_draws]
  We adds temperature annealing on top of it, which is:
    g = -gumbel(key, (n_inputs,)) - temperature * jnp.log(p)
    ind = jnp.argsort(g)[:n_draws]
  Args:
    rng: a PRNG key used as the random key.
    mask_len: the number to mask.
    probs: the probabilities associated with each entry.
    temperature: when temperature = 1.0, it's identical to jax's implementation.
      The larger this value is, the more random the masking is picked.
  Returns:
    A binary masking map [batch_size, seq_len].
  """
  confidence = jnp.log(probs) + temperature * jax.random.gumbel(
      rng, probs.shape)
  sorted_confidence = jnp.sort(confidence, axis=-1)
  # Obtains cut off threshold given the mask lengths.
  cut_off = jnp.take_along_axis(sorted_confidence, mask_len.astype(jnp.int32), 
                                axis=-1)
  # Masks tokens with lower confidence.
  masking = (confidence < cut_off)
  return masking


def decode(inputs,
           rng,
           tokens_to_logits,
           mask_token_id=-1,
           num_iter=12,
           start_iter=0,
           choice_temperature=1.0,
           mask_scheduling_method='cosine'):
  """Fast decoding for iterative generation.
  Args:
    inputs: int32 array: [batch_size, seq_length] input sequence of masked
      tokens, where the masking tokens is defined by mask_token_id.
    rng: jnp.DeviceArray: sampling random state.
    tokens_to_logits: decoder function taking single token slices and cache and
      returning logits and updated cache.
    mask_token_id: int: [Mask] token id.
    num_iter: int: default is 12.
    start_iter: int: default is 0.
    choice_temperature: float: temperature to control the randomness of masking.
    mask_scheduling_method: masking method string. See mask_schedule.py for
      details.
  Returns:
     [batch_size, num_iter, seq_length] output sequence of tokens in all
       iterations.
  """
  inputs = inputs.astype('int32')
  unknown_number_in_the_beginning = jnp.sum(inputs == mask_token_id, axis=-1)
  # Initializes state
  init_state = parallel_decode.state_init(inputs, rng, num_iter, start_iter=start_iter)

  def loop_cond_fn(state):
    """Beam search loop termination condition."""
    not_at_end = (state.cur_index < num_iter)
    return not_at_end

  def loop_body_fn(state):
    """Beam search loop state update function."""
    rng = state.rng
    step = state.cur_index
    # Current input ids: [batch_size, seq_length].
    cur_ids = state.cur_seqs

    # Calls model on current seqs to get next-iteration seqs.
    logits = tokens_to_logits(cur_ids)
    rng, sample_rng = jax.random.split(rng, 2)
    # Samples the ids using categorical sampling: [batch_size, seq_length].
    sampled_ids = jax.random.categorical(sample_rng, logits)

    # Just updates the masked tokens.
    unknown_map = (cur_ids == mask_token_id)
    sampled_ids = jnp.where(unknown_map, sampled_ids, cur_ids)
    # Defines the mask ratio for the next round. The number to mask out is
    # determined by mask_ratio * unknown_number_in_the_beginning.
    ratio = 1. * (step + 1) / num_iter
    mask_ratio = mask_schedule.schedule(ratio, unknown_number_in_the_beginning,
                                        mask_scheduling_method)
    # Updates final seqs with the current sampled_ids.
    final_seqs = jax.lax.dynamic_update_slice(
        state.final_seqs, jnp.expand_dims(sampled_ids, axis=1), (0, step, 0))
    # Computes the probabilities of each selected tokens.
    probs = jax.nn.softmax(logits, axis=-1)
    selected_probs = jnp.squeeze(
        jnp.take_along_axis(probs, jnp.expand_dims(sampled_ids, -1).astype(jnp.int32), -1), -1)
    # Ignores the tokens given in the input by overwriting their confidence.
    selected_probs = jnp.where(unknown_map, selected_probs,
                               _CONFIDENCE_OF_KNOWN_TOKENS)
    # Gets mask lens for each sample in the batch according to the mask ratio.
    mask_len = jnp.expand_dims(
        jnp.floor(unknown_number_in_the_beginning * mask_ratio), 1)
    # Keeps at least one of prediction in this round and also masks out at least
    # one and for the next iteration
    mask_len = jnp.maximum(
        1,
        jnp.minimum(jnp.sum(unknown_map, axis=-1, keepdims=True) - 1, mask_len))

    # Adds noise for randomness
    rng, choice_rng = jax.random.split(rng)
    masking = mask_by_random_topk(choice_rng, mask_len, selected_probs,
                                  choice_temperature * (1. - ratio))
    # Masks tokens with lower confidence.
    sampled_ids = jnp.where(masking, mask_token_id, sampled_ids)
    return parallel_decode.State(
        cur_index=state.cur_index + 1,
        cur_seqs=sampled_ids,
        rng=rng,
        final_seqs=final_seqs)

  # Run while loop and get final beam search state.
  final_state = jax.lax.while_loop(loop_cond_fn, loop_body_fn, init_state)
  return final_state.final_seqs

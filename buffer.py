"""
Buffer for train.py
"""
from functools import partial
import jax
from jax import numpy as jnp


def create_buffer_states(max_size, init_sample):
    buffer = [jnp.zeros((max_size, *init_sample[i].shape), init_sample[i].dtype)
              for i in range(len(init_sample))]
    insert_pos = 0
    buffer_states = {'buffer': buffer, 'insert_pos': jnp.array(insert_pos), 'max_size': jnp.array(max_size)}
    return buffer_states


@jax.jit
def put_to_buffer(buffer_state, x):
    @jax.jit
    def insert(buffer, x, position):
        for xi in range(len(x)):
            buffer[xi] = buffer[xi].at[position].set(x[xi])
        return buffer

    buffer = insert(buffer_state['buffer'], x, buffer_state['insert_pos'])
    insert_pos = (buffer_state['insert_pos'] + 1) % buffer_state['max_size']
    return dict(buffer_state, buffer=buffer, insert_pos=insert_pos)


@partial(jax.jit, static_argnums=(1,))
def sample_from_buffer(buffer_state, sample_len, key):
    # Not consider done
    # max_val = buffer_state['insert_pos'] - sample_len + buffer_state['max_size']
    # min_val = buffer_state['insert_pos']
    # begin_index = jax.random.randint(key, (1,), minval=min_val, maxval=max_val) % buffer_state['max_size']
    begin_index = jnp.zeros((1,), dtype=jnp.int8)  # fixme: take from beginning

    indices = (jnp.arange(sample_len) + begin_index) % buffer_state['max_size']
    batch = [jnp.take(buffer_state['buffer'][xi], indices, axis=0) for xi in range(len(buffer_state['buffer']))]
    batch = {'obs_embed': batch[0], 'action_embed': batch[1], 'his_hippo_hidden': batch[2], 'his_theta': batch[3],
             'his_rewards': batch[4], 'his_action': batch[5], 'his_logits': batch[6], 'his_values': batch[7]
             }
    # for k in batch:
    #     if k == 'his_rewards':  # fixme: his_rewards is of t-1, align it with other
    #         batch[k] = batch[k][1:]
    #     else:
    #         batch[k] = batch[k][:-1]
    return batch


def clear_buffer(buffer_state):
    buffer = [jnp.zeros_like(buffer_state['buffer'][xi])
              for xi in range(len(buffer_state['buffer']))]
    insert_pos = 0
    buffer_state = {'buffer': buffer, 'insert_pos': jnp.array(insert_pos), 'max_size': buffer_state['max_size']}
    return buffer_state
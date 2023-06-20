"""
Pretrain hippo module and encoder module with two task: 1. predict place cell; 2. predict reward
"""

import jax
import jax.numpy as jnp
from flax import linen as nn  # Linen API
from functools import partial
from clu import metrics
from flax.training import train_state, checkpoints  # Useful dataclass to keep train state
from flax import struct  # Flax dataclasses
import optax  # Common loss functions and optimizers
from jax import xla_computation
from tensorboardX import SummaryWriter

import env
from agent import Encoder, Hippo
import config


def create_place_cell_state(sigma, width, height):
    all_centers = []
    for i in range(height):
        for j in range(width):
            all_centers.append(jnp.array([i, j]))
    centers = jnp.stack(all_centers, axis=0)  # [m, 2]
    return {'sigma': jnp.array(sigma), 'centers': centers}


@jax.jit
def generate_place_cell(centers, sigma, x):
    # x[n, 2], centers[m, 2]
    # @partial(jax.jit, static_argnums=(2,))
    @jax.jit
    def cal_dist(pos, cents, sigma):
        # pos[2,], cents[m, 2]
        return - ((pos.reshape((1, -1)) - cents) ** 2).sum(axis=-1) / (2 * sigma ** 2)  # [m,]

    activation = jax.vmap(cal_dist, (0, None, None), 0)(x, centers, sigma)  # [n, m]
    activation = nn.softmax(activation, axis=-1)
    return activation


@struct.dataclass
class Metrics(metrics.Collection):
    loss: metrics.Average.from_output('loss')
    # acc: metrics.Average.from_output('acc')
    loss_last: metrics.Average.from_output('loss_last')
    loss_pred: metrics.Average.from_output('loss_pred')
    acc_last: metrics.Average.from_output('acc_last')
    acc_pred: metrics.Average.from_output('acc_pred')


class TrainState(train_state.TrainState):
    metrics: Metrics


def create_train_state(encoder, hippo, rng, init_sample, config):
    """Creates an initial `TrainState`."""
    # Initialize encoder ==================================================================
    rng, sub_rng = jax.random.split(rng)
    params = encoder.init(sub_rng, *init_sample)['params']
    tx = optax.adamw(config.lr, weight_decay=config.wd)
    encoder_state = TrainState.create(
        apply_fn=encoder.apply, params=params, tx=tx,
        metrics=Metrics.empty())
    # Initialize hippo ====================================================================
    obs_embed, action_embed = encoder_state.apply_fn({'params': params}, *init_sample)
    hidden = jnp.zeros((config.n_agents, config.hidden_size))
    pfc_input = jnp.zeros((config.n_agents, config.bottleneck_size))
    rng, sub_rng = jax.random.split(rng)
    params = hippo.init(sub_rng, hidden, pfc_input, (obs_embed, action_embed), jnp.zeros((config.n_agents, 1)))['params']
    tx = optax.adamw(config.lr, weight_decay=config.wd)
    hippo_state = TrainState.create(
        apply_fn=hippo.apply, params=params, tx=tx,
        metrics=Metrics.empty())
    return encoder_state, hippo_state


@partial(jax.jit, static_argnums=(3, 4, 5, 6))
def train_step(running_encoder_state, running_hippo_state, batch, sample_len, n_agent, hidden_size, bottleneck_size):
    """Train for a single step with rollouts from the buffer"""
    # state, obs(o_t)[t, n, h, w], actions(action_t-1)[t, n, 1], rewards(r_t-1)[t, n, 1], real_pos(s_t)[t, n, 2]
    # o_t, r_t-1, action_t-1 => s_t, r_t

    #  Initialize hidden
    hiddens = jnp.zeros((n_agent, hidden_size))

    def forward_fn(params_encoder, params_hippo, n_agent, bottleneck_size, hiddens, inputs):
        obs, action, rewards_prev = inputs
        obs_embed, action_embed = running_encoder_state.apply_fn({'params': params_encoder}, obs, action)
        pfc_input = jnp.zeros((n_agent, bottleneck_size))
        new_hidden, outputs = running_hippo_state.apply_fn({'params': params_hippo},
                                                           hiddens, pfc_input, (obs_embed, action_embed), rewards_prev)
        return new_hidden, outputs

    def loss_fn(params_encoder, params_hippo, hiddens, batch):
        len_t, n_agent, num_cells = batch['place_cells'].shape
        num_rewards = batch['rewards'].shape[-1]
        apply_fn = partial(forward_fn, params_encoder, params_hippo, n_agent, bottleneck_size)
        _, all_preds = jax.lax.scan(apply_fn, hiddens, [batch['obs'], batch['action'], batch['rewards']])
        preds_place = all_preds[:, :, :num_cells]
        preds_rewards = all_preds[:, :, num_cells:num_cells+num_rewards]  # [t, n, 1]
        preds_reward_distance = all_preds[:, :, num_cells+num_rewards:] # [t, n, 1]
        loss_pred = optax.softmax_cross_entropy(preds_place, batch['place_cells']).mean()
        acc_pred = (jnp.argmax(all_preds, axis=-1)
                    == jnp.argmax(batch['place_cells'], axis=-1)).astype(jnp.float32).mean()

        # pred last
        rewards_label = batch['rewards'][1:]
        loss_last = jnp.abs(preds_rewards[:-1] - rewards_label)  # [t-1, n, 1]
        loss_last = jnp.where(rewards_label > 0.4, loss_last * 20, loss_last)  # fixme: weighted loss
        acc_last = jnp.where((loss_last < 0.1) & (jnp.abs(rewards_label-config.mid_reward)<1e-5), 1, 0)
        consider_last_flag = jnp.cumsum(rewards_label, axis=0) > 0.6
        loss_last = jnp.where(consider_last_flag, loss_last, 0).mean()
        acc_last = jnp.where(consider_last_flag, acc_last, 0).sum() \
                   / jnp.where(jnp.abs(rewards_label-config.mid_reward)<1e-5 & consider_last_flag, 1, 0).sum()
        # fixme: cumsum_rewards > 0.6, considering random reward value is 0.5, > 0.6 means the second time met a reward

        loss_dist = jnp.abs(preds_reward_distance - batch['reward_distance_coding']).mean()
        acc_dist = jnp.where((jnp.abs(preds_reward_distance - batch['reward_distance_coding']) < 0.05), 1, 0).astype(jnp.float32).mean()
        loss = loss_pred + loss_last + loss_dist
        return loss, (loss_last, loss_pred, loss_dist, acc_last, acc_pred, acc_dist)

    grad_fn = jax.value_and_grad(partial(loss_fn, hiddens=hiddens, batch=batch), has_aux=True, argnums=(0, 1))
    (loss, (loss_last, loss_pred, loss_dist, acc_last, acc_pred, acc_dist)), (grads_encoder, grads_hippo) = grad_fn(
        running_encoder_state.params,
        running_hippo_state.params)
    running_encoder_state = running_encoder_state.apply_gradients(grads=grads_encoder)
    clip_fn = lambda z: jnp.clip(z, -1.0, 1.0)  # fixme: clip by value / by grad
    grads_hippo = jax.tree_util.tree_map(clip_fn, grads_hippo)
    running_hippo_state = running_hippo_state.apply_gradients(grads=grads_hippo)

    # compute metrics
    metric_updates = running_encoder_state.metrics.single_from_model_output(
        loss=loss, loss_last=loss_last, loss_pred=loss_pred, loss_dist=loss_dist, acc_last=acc_last, acc_pred=acc_pred, acc_dist=acc_dist)
    running_encoder_state = running_encoder_state.replace(metrics=metric_updates)
    return running_encoder_state, running_hippo_state


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
    max_val = buffer_state['insert_pos'] - sample_len + buffer_state['max_size']
    min_val = buffer_state['insert_pos']
    begin_index = jax.random.randint(key, (1,), minval=min_val, maxval=max_val) % buffer_state['max_size']

    indices = (jnp.arange(sample_len) + begin_index) % buffer_state['max_size']
    return [jnp.take(buffer_state['buffer'][xi], indices, axis=0) for xi in range(len(buffer_state['buffer']))]

    # return rollout

# @jax.jit
def calculate_distance_coding(current_pos, reward_pos, goal_pos, mid_reward, reward_distance_ratio):
    # [t, n, 2], [t, n, 2], [t, n, 2]
    print('current_pos', current_pos.shape)
    print('reward_pos', reward_pos.shape)
    print('goal_pos', goal_pos.shape)
    mid_distance = ((current_pos - reward_pos)**2).sum(axis=-1)**0.5
    goal_distance = ((current_pos - goal_pos)**2).sum(axis=-1)**0.5
    distance_coding = jnp.exp(-mid_distance/reward_distance_ratio)*mid_reward + \
                        jnp.exp(-goal_distance/reward_distance_ratio)*1.
    return distance_coding #[n, 1]

### to be finished
def prepare_batch(rollouts, place_cell_state, config):
    # obs [t, n, h, w], actions[t, n, 1], pos[t, n, 2], rewards[t, n, 1], reward_pos[t, n, 2]
    batch = dict()
    batch['obs'] = rollouts[0]
    batch['action'] = rollouts[1]
    batch['place_cells'] = jax.vmap(generate_place_cell, (None, None, 0), 0)(place_cell_state['centers'],
                                                                             place_cell_state['sigma'],
                                                                             rollouts[2])
    cal_dist_coding = partial(calculate_distance_coding, mid_reward=config.mid_reward,
                                reward_distance_ratio=config.reward_distance_ratio)
    batch['reward_distance_coding'] = jax.vmap(cal_dist_coding, (0, 0, 0), 0)(rollouts[2],
                                                                            rollouts[4],
                                                                            rollouts[5])
    # t * n * m
    batch['rewards'] = rollouts[3]
    return batch


@partial(jax.jit, static_argnums=(2, 3, 4))
def mask_obs(obs, key, sample_len, n_agent, visual_prob):
    # obs[t, n, h, w]
    # fixme: should by env; not mask the first step
    mask = jax.random.uniform(key, (sample_len, n_agent, 1, 1))
    mask = mask.at[0, :, :, :].set(0)
    obs = jnp.where(mask < visual_prob, obs, 0)
    return obs


@partial(jax.jit, static_argnums=(5, 6, 7, 8, 9))
def a_loop(key, buffer_states, place_cell_state, running_encoder_state, running_hippo_state,
           sample_len, n_agents, visual_prob, hidden_size, bottleneck_size, config):
    # get from buffer, train_step()
    key, subkey = jax.random.split(key)
    rollouts = sample_from_buffer(buffer_states, sample_len, subkey)
    # print(ei, len(rollouts))
    batch = prepare_batch(rollouts, place_cell_state, config)
    batch['obs'] = mask_obs(batch['obs'], key,
                            sample_len, n_agents, visual_prob)
    # print(batch['place_cells'].reshape((-1, 100)).std(axis=0).mean(), 'place cell std')

    running_encoder_state, running_hippo_state = train_step(running_encoder_state, running_hippo_state, batch,
                                                            sample_len, n_agents, hidden_size, bottleneck_size)
    return key, buffer_states, place_cell_state, running_encoder_state, running_hippo_state


def main(config):
    # Initialize logs ==============================================
    # metrics_history = {'train_loss': [], 'train_accuracy': []}
    writer = SummaryWriter(f'./logs/{config.save_name}')
    # Initialize key ================================================
    key = jax.random.PRNGKey(0)
    # Initialize env and place_cell ================================================
    key, subkey = jax.random.split(key)
    obs, env_state = env.reset(config.width, config.height, config.n_agents, config.mid_reward, subkey)
    key, subkey = jax.random.split(key)
    actions = jax.random.randint(subkey, (config.n_agents, 1), minval=0, maxval=4)  # [n, 1]
    obs, rewards, done, env_state = env.step(env_state, actions)
    # place_cell = PlaceCell(config.sigma, config.width, config.height)
    place_cell_state = create_place_cell_state(config.sigma, config.width, config.height)
    # Initialize model and training_state ============================
    encoder = Encoder()
    hippo = Hippo(output_size=place_cell_state['centers'].shape[0] + 3,
                  hidden_size=config.hidden_size)
    key, subkey = jax.random.split(key)
    running_encoder_state, running_hippo_state = create_train_state(encoder, hippo, subkey,
                                                                    (obs, actions),
                                                                    config)
    if config.load != '':
        running_encoder_state = checkpoints.restore_checkpoint(ckpt_dir=config.load, target=running_encoder_state)
        running_hippo_state = checkpoints.restore_checkpoint(ckpt_dir=config.load.replace('encoder', 'hippo'), target=running_hippo_state)
        print('load from', config.load)
    # Initialize buffer================================================
    buffer_states = create_buffer_states(max_size=config.max_size, init_sample=[obs, actions, env_state['current_pos'],
                                                                                rewards, env_state['reward_pos'], env_state['goal_pos']])

    for ei in range(config.epoch):
        key, subkey = jax.random.split(key)
        actions = jax.random.randint(subkey, (config.n_agents, 1), minval=0, maxval=4)  # [n, 1]
        obs, rewards, done, env_state = env.step(env_state, actions)
        # put_to_buffer: o_t, r_t-1, action_t-1, s_t
        buffer_states = put_to_buffer(buffer_states, [obs, actions, env_state['current_pos'],
                                                      rewards, env_state['reward_pos'], env_state['goal_pos']])
        # put to buffer [obs_t, a_t-1, pos_t, reward_t]
        key, subkey = jax.random.split(key)
        env_state = env.reset_reward(env_state, rewards, subkey)  # fixme: reset reward

        if ei % config.train_every == 0 and ei > config.max_size:
            key, buffer_states, place_cell_state, running_encoder_state, running_hippo_state = \
                a_loop(key, buffer_states, place_cell_state, running_encoder_state, running_hippo_state,
                       sample_len=config.sample_len, n_agents=config.n_agents,
                       visual_prob=config.visual_prob, hidden_size=config.hidden_size,
                       bottleneck_size=config.bottleneck_size, config=config)

        if ei % 100 == 0 and ei > config.max_size:
            for k, v in running_encoder_state.metrics.compute().items():
                print(ei, k, v.item())
                writer.add_scalar(f'train_{k}', v.item(), ei + 1)
        if ei % 5000 == 0 and ei > config.max_size:
            checkpoints.save_checkpoint(f'./modelzoo/{config.save_name}_encoder', target=running_encoder_state, step=ei)
            checkpoints.save_checkpoint(f'./modelzoo/{config.save_name}_hippo', target=running_hippo_state, step=ei)


if __name__ == '__main__':
    main(config)

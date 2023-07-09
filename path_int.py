"""
Pretrain hippo module and encoder module with two task: 1. predict place cell; 2. predict reward
"""

import jax
import jax.numpy as jnp
from flax import linen as nn  # Linen API
from functools import partial
from clu import metrics
import flax.core
from flax.training import train_state, checkpoints  # Useful dataclass to keep train state
from flax import struct  # Flax dataclasses
from flax.traverse_util import flatten_dict, unflatten_dict
import optax  # Common loss functions and optimizers
from jax import xla_computation
from tensorboardX import SummaryWriter
import os
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


# @struct.dataclass
# class Metrics(metrics.Collection):
#     loss: metrics.Average.from_output('loss')
#     # acc: metrics.Average.from_output('acc')
#     loss_last: metrics.Average.from_output('loss_last')
#     loss_pred: metrics.Average.from_output('loss_pred')
#     # acc_0: metrics.Average.from_output('acc_0')
#     # acc_r: metrics.Average.from_output('acc_r')
#     # acc_g: metrics.Average.from_output('acc_g')
#     acc_last: metrics.Average.from_output('acc_last')
#     acc_pred: metrics.Average.from_output('acc_pred')


@struct.dataclass
class Metrics(metrics.Collection):
    loss: metrics.Average.from_output('loss')
    # acc: metrics.Average.from_output('acc')
    loss_last: metrics.Average.from_output('loss_last')
    loss_pred: metrics.Average.from_output('loss_pred')
    loss_dist: metrics.Average.from_output('loss_dist')
    acc_0: metrics.Average.from_output('acc_0')
    acc_r: metrics.Average.from_output('acc_r')
    acc_g: metrics.Average.from_output('acc_g')
    # acc_last: metrics.Average.from_output('acc_last')
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
        metrics=Metrics.empty())  # todo: metrics2
    # Initialize hippo ====================================================================
    obs_embed, action_embed = encoder_state.apply_fn({'params': params}, *init_sample)
    hidden = jnp.zeros((config.n_agents, config.hidden_size))
    pfc_input = jnp.zeros((config.n_agents, config.bottleneck_size))
    rng, sub_rng = jax.random.split(rng)
    params = hippo.init(sub_rng, hidden, pfc_input, (obs_embed, action_embed), jnp.zeros((config.n_agents, 1)))[
        'params']
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
        masked_rewards = batch['rewards'].at[-1, :, :].set(0)  # todo: mask and predict the last reward
        _, all_preds = jax.lax.scan(apply_fn, hiddens, [batch['obs'], batch['action'], masked_rewards])
        preds_place = all_preds[:, :, :num_cells]
        preds_rewards = all_preds[:, :, num_cells:num_cells+num_rewards]  # [t, n, 1]
        preds_distance = all_preds[:, :, num_cells+num_rewards:]  # [t, n, 2]
        loss_dist = jnp.abs(preds_distance - batch['reward_distance']).mean()
        loss_pred = optax.softmax_cross_entropy(preds_place, batch['place_cells'])[:-1].mean()
        # exclude the last step of place_cell prediction
        # todo: [:-1]: not to pred the last place cell because of the masked rewards
        acc_pred = (jnp.argmax(preds_place, axis=-1)  # todo: fix a bug: all_preds -> preds_place
                    == jnp.argmax(batch['place_cells'], axis=-1)).astype(jnp.float32)[:-1].mean()

        # pred last
        rewards_label = batch['rewards'][-1]  # [n, 1]  # todo: 1: :-1; only predict the last one
        loss_last = jnp.abs(preds_rewards[-1] - rewards_label)  # [n, 1]  # only consider the last step
        loss_last = jnp.where(rewards_label > 0.4, loss_last * 10, loss_last)  # todo: weighted loss, times by 10
        acc_0 = jnp.where((loss_last < 0.2) & (jnp.abs(rewards_label - 0.) < 1e-3), 1, 0)  # todo: acc criterion: < 0.2
        acc_r = jnp.where((loss_last < 0.2) & (jnp.abs(rewards_label - 0.5) < 1e-3), 1, 0)
        acc_g = jnp.where((loss_last < 0.2) & (jnp.abs(rewards_label - 1.) < 1e-3), 1, 0)
        consider_last_flag = (jnp.sum(
            jnp.where(jnp.abs(batch['rewards'] - 1) < 1e-3, 0, batch['rewards']), axis=0) > 0.6
                              ) | (rewards_label > 0.9)  # [n, 1], todo: consider: 1. gets reward twice; or 2. get goal
        # jax.debug.print('{a}_{b}', a=(jnp.sum(
        #     jnp.where(jnp.abs(batch['rewards'] - 1) < 1e-3, 0, batch['rewards']), axis=0) > 0.6
        #                               ).mean(), b=(rewards_label > 0.9).mean())
        jnp.set_printoptions(threshold=jnp.inf)
        jax.debug.print('rewards {shape} be like:{a}',a=batch['rewards'][:,0,:].reshape(-1,),shape=batch['rewards'].shape[0])
        jax.debug.print('consider_last_flag:{flag}',flag=consider_last_flag)
        loss_last = jnp.where(consider_last_flag, loss_last, 0).mean()
        acc_0 = jnp.where(consider_last_flag, acc_0, 0).sum() \
                / jnp.where((jnp.abs(rewards_label - 0.) < 1e-3) & consider_last_flag, 1, 0).sum()
        acc_r = jnp.where(consider_last_flag, acc_r, 0).sum() \
                / jnp.where((jnp.abs(rewards_label - 0.5) < 1e-3) & consider_last_flag, 1, 0).sum()
        acc_g = jnp.where(consider_last_flag, acc_g, 0).sum() \
                / jnp.where((jnp.abs(rewards_label - 1.) < 1e-3) & consider_last_flag, 1, 0).sum()
        # jax.debug.print('direct_compare_acc_r_nominator:{a}, direct_compare_acc_g_nominator:{b}',\
        #     a=jnp.where((jnp.abs(rewards_label - 0.5) < 1e-3) & consider_last_flag, 1, 0).sum(),\
        #     b=jnp.where((jnp.abs(rewards_label - 1.) < 1e-3) & consider_last_flag, 1, 0).sum())
        # jax.debug.print('isclose_acc_r_nominator:{a}, isclose_acc_g_nominator:{b}',\
        #     a=jnp.where(jnp.isclose(rewards_label, 0.5) & consider_last_flag, 1, 0).sum(),\
        #     b=jnp.where(jnp.isclose(rewards_label, 1.) & consider_last_flag, 1, 0).sum())
        # fixme: cumsum_rewards > 0.6, considering random reward value is 0.5, > 0.6 means the second time met a reward

        loss = loss_pred + loss_last * 0.5 + loss_dist  # todo: *0.1
        return loss, (loss_last, loss_pred, loss_dist, acc_0, acc_r, acc_g, acc_pred)

    grad_fn = jax.value_and_grad(partial(loss_fn, hiddens=hiddens, batch=batch), has_aux=True, argnums=(0, 1))
    (loss, (loss_last, loss_pred, loss_dist, acc_0, acc_r, acc_g, acc_pred)), (grads_encoder, grads_hippo) = grad_fn(
        running_encoder_state.params,
        running_hippo_state.params)
    running_encoder_state = running_encoder_state.apply_gradients(grads=grads_encoder)
    # todo: clip by value / by grad ============================
    # clip_fn = lambda z: jnp.clip(z, -1.0, 1.0)
    # grads_hippo = jax.tree_util.tree_map(clip_fn, grads_hippo)
    # ----------------------------------------------------------
    grads_hippo = flatten_dict(grads_hippo).copy()
    jax.debug.print('grad_{a}', a=jnp.linalg.norm(grads_hippo[('Dense_0', 'kernel')], ord=2))
    grads_hippo[('Dense_0', 'kernel')] \
        = grads_hippo[('Dense_0', 'kernel')] / jnp.maximum(jnp.linalg.norm(grads_hippo[('Dense_0', 'kernel')], ord=2), 5.0) * 5
    grads_hippo = unflatten_dict(grads_hippo)
    grads_hippo = flax.core.freeze(grads_hippo)
    # ==========================================================
    running_hippo_state = running_hippo_state.apply_gradients(grads=grads_hippo)

    # compute metrics
    metric_updates = running_encoder_state.metrics.single_from_model_output(
        loss=loss, loss_last=loss_last, loss_pred=loss_pred, loss_dist=loss_dist, acc_0=acc_0, acc_r=acc_r, acc_g=acc_g, acc_pred=acc_pred)
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


# @partial(jax.jit, static_argnums=(1,2))
def sample_from_buffer(buffer_state, sample_len, n_agents, key):
    # Not consider done
    max_val = buffer_state['insert_pos'] - sample_len + buffer_state['max_size']
    min_val = buffer_state['insert_pos']
    ### 改成agent数目
    begin_index = jax.random.randint(key, (n_agents,1), minval=min_val, maxval=max_val) % buffer_state['max_size']
    # n_agent * sample_len
    indices = (jnp.arange(sample_len).reshape(1,-1).repeat(n_agents,0) + begin_index.repeat(sample_len,axis=1)) % buffer_state['max_size']
    # buffer: xi * buffer_size * n_agents * xi_specific_dimension
    # jax.debug.print('indices:{a}',a=indices)
    samples = [[] for _ in range(len(buffer_state['buffer']))]
    for xi in range(len(buffer_state['buffer'])):
        for agent_th in range(n_agents):
            # jax.debug.print(f"buffer_shape:{buffer_state['buffer'][xi].shape}")
            samples[xi].append(buffer_state['buffer'][xi][indices[agent_th], agent_th])
        samples[xi] = jnp.stack(samples[xi], axis=1)
        # jax.debug.print(f'samples:{samples[xi].shape}')
    # jax.debug.breakpoint()
    return samples




def prepare_batch(rollouts, place_cell_state):
    # obs [t, n, h, w], actions[t, n, 1], pos[t, n, 2], rewards[t, n, 1]
    # reward_pos [t, n, 2] goal_pos [t, n, 2]
    # t = sample_len
    batch = dict()
    batch['obs'] = rollouts[0]
    batch['action'] = rollouts[1]
    batch['place_cells'] = jax.vmap(generate_place_cell, (None, None, 0), 0)(place_cell_state['centers'],
                                                                             place_cell_state['sigma'],
                                                                             rollouts[2])
    batch['rewards'] = rollouts[3]
    mid_reward_distance = jnp.exp(-jnp.sqrt(jnp.sum((rollouts[4]-rollouts[2])**2,axis=-1)))
    goal_distance = jnp.exp(-jnp.sqrt(jnp.sum((rollouts[5]-rollouts[2])**2,axis=-1)))
    batch['reward_distance'] = jnp.stack((mid_reward_distance, goal_distance),axis=-1)
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
           sample_len, n_agents, visual_prob, hidden_size, bottleneck_size):
    # get from buffer, train_step()
    key, subkey = jax.random.split(key)
    rollouts = sample_from_buffer(buffer_states, sample_len, n_agents, subkey)
    # print(ei, len(rollouts))
    batch = prepare_batch(rollouts, place_cell_state)
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
    obs, env_state = env.reset(config.width, config.height, config.n_agents, subkey)
    key, subkey = jax.random.split(key)
    actions = jax.random.randint(subkey, (config.n_agents, 1), minval=0, maxval=4)  # [n, 1]
    obs, rewards, done, env_state = env.step(env_state, actions)
    # place_cell = PlaceCell(config.sigma, config.width, config.height)
    place_cell_state = create_place_cell_state(config.sigma, config.width, config.height)
    # Initialize model and training_state ============================
    encoder = Encoder()
    hippo = Hippo(output_size=place_cell_state['centers'].shape[0] + 1 + 2,
                  hidden_size=config.hidden_size)
    key, subkey = jax.random.split(key)
    running_encoder_state, running_hippo_state = create_train_state(encoder, hippo, subkey,
                                                                    (obs, actions),
                                                                    config)
    if config.load != '':
        if config.load[2:] in os.listdir():
            print('successfully load from:', config.load)
        else:
            print('randomly initialized')
        running_encoder_state = checkpoints.restore_checkpoint(ckpt_dir=config.load, target=running_encoder_state)
        running_hippo_state = checkpoints.restore_checkpoint(ckpt_dir=config.load.replace('encoder', 'hippo'),
                                                             target=running_hippo_state)
        # running_encoder_state = running_encoder_state.replace(metrics=Metrics2.empty())
    # Initialize buffer================================================
    buffer_states = create_buffer_states(max_size=config.max_size, init_sample=[obs, actions, env_state['current_pos'],
                                                                                rewards, env_state['goal_pos'], env_state['reward_center']])

    for ei in range(config.epoch):
        key, subkey = jax.random.split(key)
        actions = jax.random.randint(subkey, (config.n_agents, 1), minval=0, maxval=4)  # [n, 1]
        obs, rewards, done, env_state = env.step(env_state, actions)
        # put_to_buffer: o_t, r_t-1, action_t-1, s_t
        buffer_states = put_to_buffer(buffer_states, [obs, actions, env_state['current_pos'],
                                                      rewards, env_state['goal_pos'], env_state['reward_center']])
        # put to buffer [obs_t, a_t-1, pos_t, reward_t]
        key, subkey = jax.random.split(key)
        env_state = env.reset_reward(env_state, rewards, subkey)  # fixme: reset reward

        if ei % config.train_every == 0 and ei > config.max_size:
            key, buffer_states, place_cell_state, running_encoder_state, running_hippo_state = \
                a_loop(key, buffer_states, place_cell_state, running_encoder_state, running_hippo_state,
                       sample_len=config.sample_len, n_agents=config.n_agents,
                       visual_prob=config.visual_prob, hidden_size=config.hidden_size,
                       bottleneck_size=config.bottleneck_size)

        if ei % 100 == 0 and ei > config.max_size:
            for k, v in running_encoder_state.metrics.compute().items():
                print(ei, k, v.item())
                if jnp.isnan(v).item():
                    print(k, v)  # fixme
                else:
                    writer.add_scalar(f'train_{k}', v.item(), ei + 1) 
        if ei % 5000 == 0 and ei > config.max_size:
            checkpoints.save_checkpoint(f'./modelzoo/{config.save_name}_encoder', target=running_encoder_state, step=ei, overwrite=True)
            checkpoints.save_checkpoint(f'./modelzoo/{config.save_name}_hippo', target=running_hippo_state, step=ei, overwrite=True)


if __name__ == '__main__':
    main(config)


### mask Reward 预测最后一个
### flag 只考虑同时有两个reward的情况
##### 交替训练？### Fine-tune? 后边再试 现在不太行
### VAE : 先试试能不能过拟合，以及滤波之后的低维

### 阻断replay后再进行学习
### 学习一些生物故事
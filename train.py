"""
Main module with PPO navigation task
"""
import argparse
from functools import partial
import jax
from jax import numpy as jnp
import optax
from flax import struct  # Flax dataclasses
from clu import metrics
from tensorboardX import SummaryWriter

import env
from agent import Encoder, Hippo, Policy
from flax.training import train_state, checkpoints
import path_int
import buffer
import os 

def parse_args():
    # args.
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--wd', type=float, default=0.0)
    parser.add_argument('--train_every', type=int, default=1000)
    parser.add_argument('--n_agents', type=int, default=128)
    parser.add_argument('--max_size', type=int, default=1024)  # max_size of buffer
    parser.add_argument('--sample_len', type=int, default=1000)  # sample len from buffer: at most max_size - 1
    parser.add_argument('--epochs', type=int, default=int(1e6))

    parser.add_argument('--save_name', type=str, default='train0')
    parser.add_argument('--model_path', type=str, default='./modelzoo')

    parser.add_argument('--mid_reward', type=float, default=2)
    parser.add_argument('--replay_steps', type=int, default=8)  # todo: tune

    parser.add_argument('--gamma', type=float, default=0.95)
    parser.add_argument('--clip_param', type=float, default=0.2)
    parser.add_argument('--entropy_coef', type=float, default=1e-2)
    parser.add_argument('--n_train_time', type=int, default=16)

    # params that should be the same with config.py
    parser.add_argument('--bottleneck_size', type=int, default=8)
    parser.add_argument('--width', type=int, default=10)
    parser.add_argument('--height', type=int, default=10)
    parser.add_argument('--n_action', type=int, default=4)
    parser.add_argument('--visual_prob', type=float, default=0.05)
    parser.add_argument('--load_encoder', type=str, default='./modelzoo/r_input_encoder/checkpoint_995000')  # todo: checkpoint
    parser.add_argument('--load_hippo', type=str, default='./modelzoo/r_input_hippo/checkpoint_995000')
    parser.add_argument('--load_policy', type=str, default='./modelzoo/r_policy_huge_reward995001')

    parser.add_argument('--hidden_size', type=int, default=128)
    args = parser.parse_args()
    return args


@jax.jit
def sample_from_policy(logit, key, temperature):
    # logit [n, 4]
    # return action [n, 1]
    def sample_once(logit_n, subkey):
        # logit_n[4,]
        action_n = jax.random.choice(subkey, jnp.arange(0, logit_n.shape[-1]), shape=(1,),
                                     p=jax.nn.softmax(logit_n / temperature, axis=-1))
        return action_n

    subkeys = jax.random.split(key, num=logit.shape[0])
    subkeys = jnp.stack(subkeys, axis=0)
    action = jax.vmap(sample_once, 0, 0)(logit, subkeys).astype(jnp.int8)
    return action


@partial(jax.jit, static_argnums=(2, 3, 4, 5, 6, 7, 8))
def train_step(state, batch, sample_len, n_agents, hidden_size, replay_steps, clip_param, entropy_coef, n_train_time):
    """Train for a single step with rollouts from the buffer, update policy_state only"""
    # state (encoder_state, hippo_state, policy_state)
    # obs_emb_t[t, n, d], action_emb_t-1, h_t[t, n, h] (before replay), theta_t[t, n, h] (before replay)，
    # rewards_t[t, n, 1], action_t[t, n, 1], policy_t[t, n, 4], value_t[t, n, 1]
    # traced_rewards_t[t, n, 1]: exp avg of rewards_t (MC sample of true value)
    # all rollouts start from where with rewards or reset, and first obs is not zero (optional)
    # his means that the data is from buffer (history)

    encoder_state, hippo_state, policy_state = state

    def loss_fn(policy_params, batch):
        # his_theta [t, n, h]
        # obs_embed [t, n, 48], action_embed [t, n, 4]
        # his_action [t, n, 1], his_logits [t, n, 4], his_rewards [t, n, 1], his_values[t, n, 1]
        # his_traced_rewards [t, n, 1]

        # 1. Replay: generate his_replayed_theta =======================================================
        replay_fn_to_scan = partial(replay_fn_for_seq, policy_params=policy_params,
                                    hippo_state=hippo_state, policy_state=policy_state,
                                    n_agents=n_agents,
                                    obs_embed_size=batch['obs_embed'].shape[-1],
                                    action_embed_size=batch['action_embed'].shape[-1])
        (_, his_replayed_theta), _ = jax.lax.scan(replay_fn_to_scan,
                                                  init=(batch['his_hippo_hidden'], batch['his_theta']),
                                                  xs=None, length=replay_steps)

        def propagate_theta(theta_step, reward_and_theta):
            # [n, h], [n, 1], [n, h]
            reward_step, replayed_theta_step = reward_and_theta
            theta_step = jnp.where(reward_step > 0, replayed_theta_step, theta_step)
            return theta_step, theta_step

        _, his_replayed_theta = jax.lax.scan(propagate_theta, init=batch['his_theta'][0], xs=(batch['his_rewards'],
                                                                                              his_replayed_theta))
        # todo: propagate theta
        # his_replayed_theta = jnp.where(batch['his_rewards'] > 0, his_replayed_theta, batch['his_theta'])

        # 2. Take action ===================================================================================
        def forward_fn1(theta_t, obs_embed_t, hippo_hidden_t):
            # theta[n, h]; obs[n, 48]; hippo_hidden_t[n, h]
            _, (policy_t, value_t, _) = policy_state.apply_fn({'params': policy_params},
                                                              theta_t, obs_embed_t,
                                                              hippo_hidden_t)
            return policy_t, value_t  # [n, 4], [n, 1]

        policy_logits, value = jax.vmap(forward_fn1, (0, 0, 0), (0, 0))(his_replayed_theta,
                                                                        batch['obs_embed'],
                                                                        jnp.zeros_like(batch['his_hippo_hidden']))
        # the pfc cannot see hippo during walk
        # policy_logits_t[t, n, 4], value_t[t, n, 1]

        # 3. PPO =========================================================================================
        ratio = jnp.exp(
            jax.nn.log_softmax(policy_logits, axis=-1) - jax.nn.log_softmax(batch['his_logits'], axis=-1))  # [t, n, 4]
        t, n, _ = ratio.shape
        index_t = jnp.repeat(jnp.arange(0, t).reshape((t, 1)), repeats=n, axis=-1).reshape((t, n, 1))
        index_n = jnp.repeat(jnp.arange(0, n).reshape((1, n)), repeats=t, axis=0).reshape((t, n, 1))
        index_action = jnp.concatenate((index_t, index_n, batch['his_action']), axis=-1)  # [n, 3]
        ratio = jax.lax.gather(ratio, start_indices=index_action,
                               dimension_numbers=jax.lax.GatherDimensionNumbers(offset_dims=(2,),
                                                                                collapsed_slice_dims=(0, 1),
                                                                                start_index_map=(0, 1, 2)),
                               slice_sizes=(1, 1, 1))  # [t, n, 1]  # fixme: how to use gather in jax? maybe bugs
        # debug variables ------------------------------------
        # # if n_train == 0:  # fixme
        # #     assert ((ratio < 1 + 1e-3) & (ratio > 1 - 1e-3)).float().mean() > 0.999, (ratio.max(), ratio.min())
        approx_kl = ((ratio - 1) - jnp.log(ratio)).mean()
        # -----------------------------------------------------
        advantage = batch['his_traced_rewards'] - batch['his_values']  # [t, n, 1]
        surr1 = ratio * advantage
        surr2 = jnp.clip(ratio, 1.0 - clip_param,
                         1.0 + clip_param) * advantage
        action_loss = -jnp.minimum(surr1, surr2).mean()
        entropy_loss = - (jax.nn.log_softmax(policy_logits, axis=-1) * jax.nn.softmax(policy_logits, axis=-1)).sum(
            axis=-1).mean()
        value_loss = ((value - batch['his_traced_rewards']) ** 2).mean()

        loss = action_loss - entropy_loss * entropy_coef + 0.5 * value_loss

        return loss, (action_loss, entropy_loss, value_loss, approx_kl)

    for _ in range(n_train_time):
        grad_fn = jax.value_and_grad(partial(loss_fn, batch=batch), has_aux=True)
        (loss, (action_loss, entropy_loss, value_loss, approx_kl)), grad = grad_fn(policy_state.params)

        clip_fn = lambda z: jnp.clip(z, -5.0, 5.0)  # fixme: clip by value / by grad
        grad = jax.tree_util.tree_map(clip_fn, grad)
        policy_state = policy_state.apply_gradients(grads=grad)

    # compute metrics
    metric_updates = policy_state.metrics.single_from_model_output(
        loss=loss, action_loss=action_loss, entropy_loss=entropy_loss, value_loss=value_loss, approx_kl=approx_kl)
    policy_state = policy_state.replace(metrics=metric_updates)

    return policy_state


@struct.dataclass
class Metrics(metrics.Collection):
    loss: metrics.Average.from_output('loss')
    action_loss: metrics.Average.from_output('action_loss')
    entropy_loss: metrics.Average.from_output('entropy_loss')
    value_loss: metrics.Average.from_output('value_loss')
    approx_kl: metrics.Average.from_output('approx_kl')


class TrainState(train_state.TrainState):
    metrics: Metrics


def init_states(args, key, random_reset=False):
    key, subkey = jax.random.split(key)
    obs, env_state = env.reset(args.width, args.height, args.n_agents, args.mid_reward, subkey)

    # Load encoder =================================================================================
    key, subkey = jax.random.split(key)
    encoder = Encoder()
    init_samples = [jnp.zeros((args.n_agents, args.height, args.width), dtype=jnp.int8),
                    jnp.zeros((args.n_agents, 1), dtype=jnp.int8)]
    params = encoder.init(subkey, *init_samples)['params']

    running_encoder_state = path_int.TrainState.create(
        apply_fn=encoder.apply, params=params, tx=optax.adamw(0.0, weight_decay=0.0),
        metrics=path_int.Metrics.empty())
    if not random_reset:
        running_encoder_state = checkpoints.restore_checkpoint(ckpt_dir=args.load_encoder,
                                                            target=running_encoder_state)
    # Load Hippo ===========================================================================
    obs_embed, action_embed = running_encoder_state.apply_fn({'params': params}, *init_samples)
    key, subkey = jax.random.split(key)
    hippo = Hippo(output_size=args.height * args.width + 1,
                  hidden_size=args.hidden_size)
    hidden = jnp.zeros((args.n_agents, args.hidden_size))
    pfc_input = jnp.zeros((args.n_agents, args.bottleneck_size))
    params = hippo.init(subkey, hidden, pfc_input, (obs_embed, action_embed), jnp.zeros((args.n_agents, 1)))['params']

    running_hippo_state = path_int.TrainState.create(
        apply_fn=hippo.apply, params=params, tx=optax.adamw(0.0, weight_decay=0.0),
        metrics=path_int.Metrics.empty())
    if not random_reset:
        running_hippo_state = checkpoints.restore_checkpoint(ckpt_dir=args.load_hippo,
                                                            target=running_hippo_state)
    # todo: make sure the load is successful
    # Init policy state ===============================================================================
    policy = Policy(args.n_action, args.hidden_size, args.bottleneck_size)
    key, subkey = jax.random.split(key)
    theta = jnp.zeros((args.n_agents, args.hidden_size))
    params = policy.init(subkey, theta, obs_embed, hidden)['params']
    running_policy_state = TrainState.create(
        apply_fn=policy.apply, params=params, tx=optax.adamw(args.lr, weight_decay=args.wd),
        metrics=Metrics.empty())
    if not random_reset:
        running_policy_state = checkpoints.restore_checkpoint(ckpt_dir=args.load_policy,
                                                              target=running_policy_state)
    # ===============================================
    init_samples_for_buffer = [
        obs_embed, action_embed, hidden, theta,
        jnp.zeros((args.n_agents, 1)), jnp.zeros((args.n_agents, 1), dtype=jnp.int8),
        jnp.zeros((args.n_agents, args.n_action)), jnp.zeros((args.n_agents, 1))
    ]

    buffer_state = buffer.create_buffer_states(args.max_size, init_samples_for_buffer)
    return env_state, buffer_state, running_encoder_state, running_hippo_state, running_policy_state


@partial(jax.jit, static_argnums=(5, 6, 7))
def replay_fn(hippo_and_theta, xs, policy_params, hippo_state, policy_state,
              n_agents, obs_embed_size, action_embed_size):
    # to match the input/output stream of jax.lax.scan
    # and also need to calculate grad of policy_params
    hippo_hidden, theta = hippo_and_theta
    new_theta, (policy, value, to_hipp) = policy_state.apply_fn({'params': policy_params},
                                                                theta, jnp.zeros((n_agents, obs_embed_size)),
                                                                hippo_hidden)
    new_hippo_hidden, _ = hippo_state.apply_fn({'params': hippo_state.params},
                                               hippo_hidden, to_hipp,
                                               (jnp.zeros((n_agents, obs_embed_size)),
                                                jnp.zeros((n_agents, action_embed_size))),
                                               jnp.zeros((n_agents, 1)))
    return (new_hippo_hidden, new_theta), (new_hippo_hidden, new_theta)


@partial(jax.jit, static_argnums=(5, 6, 7))
def replay_fn_for_seq(hippo_and_theta, xs, policy_params, hippo_state, policy_state,
                      n_agents, obs_embed_size, action_embed_size):
    # the only difference with replay_fn is that this func replays for whole seq [t, ..., ...]
    # to match the input/output stream of jax.lax.scan
    # and also need to calculate grad of policy_params
    hippo_hidden, theta = hippo_and_theta
    new_theta, (policy, value, to_hipp) = policy_state.apply_fn({'params': policy_params},
                                                                theta, jnp.zeros(
            (hippo_hidden.shape[0], n_agents, obs_embed_size)),
                                                                hippo_hidden)
    new_hippo_hidden, _ = hippo_state.apply_fn({'params': hippo_state.params},
                                               hippo_hidden, to_hipp,
                                               (jnp.zeros((hippo_hidden.shape[0], n_agents, obs_embed_size)),
                                                jnp.zeros((hippo_hidden.shape[0], n_agents, action_embed_size))),
                                               jnp.zeros((hippo_hidden.shape[0], n_agents, 1)))
    return (new_hippo_hidden, new_theta), (new_hippo_hidden, new_theta)


@partial(jax.jit, static_argnums=(9, 10, 11, 12, 13, 14, 15))
def model_step(env_state, buffer_state, encoder_state, hippo_state, policy_state,
               key, actions, hippo_hidden, theta,
               n_agents, bottleneck_size, replay_steps, height, width, visual_prob, temperature):
    # Input: actions_t-1, h_t-1, theta_t-1,
    obs, rewards, done, env_state = env.step(env_state, actions)  # todo: reset
    key, subkey = jax.random.split(key)
    env_state = env.reset_reward(env_state, rewards, subkey)  # fixme: reset reward with 0.9 prob
    # Mask obs ==========================================================================================
    key, subkey = jax.random.split(key)
    mask = jax.random.uniform(subkey, (obs.shape[0], 1, 1))
    obs = jnp.where(mask < visual_prob, obs, 0)
    # obs[n, h, w], actions[n, 1], rewards[n, 1]
    # Encode obs and a_t-1 ===============================================================================
    obs_embed, action_embed = encoder_state.apply_fn({'params': encoder_state.params}, obs, actions)

    # Update hippo_hidden ==================================================================================
    new_hippo_hidden, _ = hippo_state.apply_fn({'params': hippo_state.params},
                                               hippo_hidden, jnp.zeros((n_agents, bottleneck_size)),
                                               (obs_embed, action_embed), rewards)

    # Replay, only when rewards > 0 ===============================================================
    replay_fn_to_scan = partial(replay_fn, policy_params=policy_state.params,
                                hippo_state=hippo_state, policy_state=policy_state,
                                n_agents=n_agents,
                                obs_embed_size=obs_embed.shape[-1], action_embed_size=action_embed.shape[-1])
    (replayed_hippo_hidden, replayed_theta), _ = jax.lax.scan(replay_fn_to_scan, init=(new_hippo_hidden, theta),
                                                              xs=None, length=replay_steps)
    # replayed_hippo_hidden = jnp.where(rewards > 0, replayed_hippo_hidden, new_hippo_hidden)
    # fixme: not save replayed_hippo_hidden
    replayed_theta = jnp.where(rewards > 0, replayed_theta, theta)

    # Take action ==================================================================================
    _, (policy, value, _) = policy_state.apply_fn({'params': policy_state.params},
                                                  replayed_theta, obs_embed, jnp.zeros_like(hippo_hidden))
    key, subkey = jax.random.split(key)
    # new_actions = jnp.argmax(policy, axis=-1, keepdims=True)
    new_actions = sample_from_policy(policy, subkey, temperature)
    # todo: reset reward; consider the checkpoint logic of env
    buffer_state = buffer.put_to_buffer(buffer_state,
                                        [obs_embed, action_embed, new_hippo_hidden, theta,
                                         rewards, new_actions, policy, value])
    # put to Buffer:
    # obs_emb_t, action_emb_t-1, h_t (before replay), theta_t (before replay)，
    # rewards_t-1, action_t, policy_t, value_t
    # jax.debug.print('obs{a}_actions_{b}_theta_{c}_rewards_{d}_newa_{e}',
    #                 a=env_state['current_pos'][0], b=actions[0], c=theta.mean(), d=rewards[0],
    #                 e=new_actions[0])
    return env_state, buffer_state, new_actions, new_hippo_hidden, replayed_theta, rewards
    # return action_t, h_t, theta_t (after replay), rewards_t-1 (for logging)


def eval_steps(env_state, buffer_state, running_encoder_state, running_hippo_state, running_policy_state,
               key, actions, hippo_hidden, theta,
               n_agents, bottleneck_size, replay_steps, height, width,
               visual_prob, n_eval_steps):
    all_rewards = []
    for _ in range(n_eval_steps):
        key, subkey = jax.random.split(key)
        env_state, buffer_state, actions, hippo_hidden, theta, rewards \
            = model_step(env_state, buffer_state, running_encoder_state, running_hippo_state, running_policy_state,
                         subkey, actions, hippo_hidden, theta,
                         n_agents, bottleneck_size, replay_steps, height, width,
                         visual_prob, temperature=0.05)
        all_rewards.append(rewards.mean().item())
    return sum(all_rewards) / len(all_rewards)


@partial(jax.jit, static_argnums=(1,))
def trace_back(rewards, gamma):
    # rewards [t, n, 1], carry, y = f(carry, x)
    t, n, _ = rewards.shape
    def trace_a_step(v, r):
        # v[n, 1], r[n, 1]
        v_prime = v * gamma + r
        return v_prime, v_prime

    def trace_gamma(gn, xs):
        # gn [1, 1]
        gn_prime = gn * gamma + 1
        return gn_prime, gn_prime

    _, all_v = jax.lax.scan(trace_a_step, jnp.zeros((n, 1)), jnp.flip(rewards, axis=0))
    _, exp_gamma = jax.lax.scan(trace_gamma, jnp.zeros((1, 1)), xs=None, length=t)
    all_v = jnp.flip(all_v, axis=0) / jnp.flip(exp_gamma, axis=0)
    return all_v


def main(args):
    key = jax.random.PRNGKey(0)
    key, subkey = jax.random.split(key)
    env_state, buffer_state, running_encoder_state, running_hippo_state, running_policy_state = init_states(args,
                                                                                                            subkey)
    writer = SummaryWriter(f'./train_logs/{args.save_name}')
    # Initialize actions, hippo_hidden, and theta ==================
    actions = jnp.zeros((args.n_agents, 1), dtype=jnp.int8)
    hippo_hidden = jnp.zeros((args.n_agents, args.hidden_size))
    theta = jnp.zeros((args.n_agents, args.hidden_size))
    # ===============================================================
    
    if args.model_path[2:] not in os.listdir():
        os.mkdir(args.model_path)
    for ei in range(args.epochs):
        # walk in the env and update buffer (model_step)
        key, subkey = jax.random.split(key)
        env_state, buffer_state, actions, hippo_hidden, theta, rewards \
            = model_step(env_state, buffer_state, running_encoder_state, running_hippo_state, running_policy_state,
                         subkey, actions, hippo_hidden, theta,
                         args.n_agents, args.bottleneck_size, args.replay_steps, args.height, args.width,
                         args.visual_prob, temperature=1.)
        if ei % 10 == 0:
            writer.add_scalar('train_reward', rewards.mean().item(), ei + 1)

        if ei % args.train_every == args.train_every - 1 and ei > args.max_size:
            # train for a step and empty buffer
            print(ei)
            key, subkey = jax.random.split(key)
            batch = buffer.sample_from_buffer(buffer_state, args.sample_len, subkey)
            batch['his_traced_rewards'] = trace_back(batch['his_rewards'], args.gamma)
            # for k in batch:
            #     print(k, batch[k].shape)
            running_policy_state = train_step((running_encoder_state, running_hippo_state, running_policy_state),
                                              batch,
                                              args.sample_len, args.n_agents, args.hidden_size, args.replay_steps,
                                              args.clip_param, args.entropy_coef, args.n_train_time)
            buffer_state = buffer.clear_buffer(buffer_state)
        if ei % 5000 == 0 and ei > args.max_size:
            key, subkey = jax.random.split(key)
            eval_rewards = eval_steps(env_state, buffer_state, running_encoder_state, running_hippo_state,
                                      running_policy_state,
                                      subkey, actions, hippo_hidden, theta,
                                      args.n_agents, args.bottleneck_size, args.replay_steps, args.height, args.width,
                                      args.visual_prob, n_eval_steps=1000)
            writer.add_scalar(f'eval_reward', eval_rewards, ei + 1)
            for k, v in running_policy_state.metrics.compute().items():
                print(k, v.item())
                writer.add_scalar(f'train_{k}', v.item(), ei + 1)
            # save model
            checkpoints.save_checkpoint(args.model_path, running_policy_state, ei + 1, prefix='r_policy', overwrite=True)


if __name__ == '__main__':
    args = parse_args()
    main(args)

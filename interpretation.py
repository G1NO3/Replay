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
import matplotlib.pyplot as plt
import train
import numpy as np
from sklearn.decomposition import PCA
import matplotlib as mpl

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
    (replayed_hippo_hidden, replayed_theta), replayed_history = jax.lax.scan(replay_fn_to_scan, init=(new_hippo_hidden, theta),
                                                              xs=None, length=replay_steps)
    # replayed_hippo_hidden = jnp.where(rewards > 0, replayed_hippo_hidden, new_hippo_hidden)
    # fixme: not save replayed_hippo_hidden
    replayed_theta = jnp.where(rewards > 0, replayed_theta, theta)

    # Take action ==================================================================================
    _, (policy, value, _) = policy_state.apply_fn({'params': policy_state.params},
                                                  replayed_theta, obs_embed, jnp.zeros_like(hippo_hidden))
###简单的假设 决策时不更新theta
    key, subkey = jax.random.split(key)
    # new_actions = jnp.argmax(policy, axis=-1, keepdims=True)
    new_actions = train.sample_from_policy(policy, subkey, temperature)
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
    return env_state, buffer_state, new_actions, new_hippo_hidden, replayed_theta, rewards, done, replayed_history
    # return action_t, h_t, theta_t (after replay), rewards_t-1 (for logging)
def replay_fn(hippo_and_theta, xs, policy_params, hippo_state, policy_state,
              n_agents, obs_embed_size, action_embed_size):
    # to match the input/output stream of jax.lax.scan
    # and also need to calculate grad of policy_params
    hippo_hidden, theta = hippo_and_theta
    new_theta, (policy, value, to_hipp) = policy_state.apply_fn({'params': policy_params},
                                                                theta, jnp.zeros((n_agents, obs_embed_size)),
                                                                hippo_hidden)
    new_hippo_hidden, output = hippo_state.apply_fn({'params': hippo_state.params},
                                               hippo_hidden, to_hipp,
                                               (jnp.zeros((n_agents, obs_embed_size)),
                                                jnp.zeros((n_agents, action_embed_size))),
                                               jnp.zeros((n_agents, 1)))
    return (new_hippo_hidden, new_theta), (new_hippo_hidden, new_theta, output)

def set_pos(grid, pos, value):
    grid = grid.at[pos[0], pos[1].set(value)]
    return grid, grid
def integrate(grid,goal_pos,hist_pos,hist_reward_pos):
    integrate_fn = partial(set_pos, value=3)
    last, trajectory = jax.lax.scan(integrate_fn, grid, hist_pos)
    trajectory = jax.vmap(set_pos, (0, hist_reward_pos, None), 0)(trajectory, hist_reward_pos, 2)

    for pos in hist_reward_pos:
        trajectory = trajectory.at[pos[0],pos[1]].set(2)
    for hpos in hist_pos:
        # print(hpos)
        trajectory = trajectory.at[hpos[0],hpos[1]].set(3)
    
    trajectory = trajectory.at[goal_pos[0],goal_pos[1]].set(4)
    print(trajectory)
    return trajectory

### reward位置不变
def main(args):
    key = jax.random.PRNGKey(0)
    key, subkey = jax.random.split(key)
    env_state, buffer_state, running_encoder_state, running_hippo_state, running_policy_state =\
         train.init_states(args, subkey, random_reset=False)
    actions = jnp.zeros((args.n_agents, 1), dtype=jnp.int32)
    hippo_hidden = jnp.zeros((args.n_agents, args.hidden_size))
    theta = jnp.zeros((args.n_agents, args.hidden_size))

    hist_pos = [[] for n in range(args.n_agents)]
    hist_reward_pos = [[] for n in range(args.n_agents)]
    hist_traj = [[] for _ in range(args.n_agents)]
    hist_replay_place = [[] for _ in range(args.n_agents)]
    hist_actions = [[] for _ in range(args.n_agents)]
    hist_hippo = []
    reward_hippo = [[],[]]
    # 0 for mid-path reward replay and 1 for goal replay
    
    plt.figure()
    for ei in range(args.epochs):
        # walk in the env and update buffer (model_step)
        if ei%30==0:
            print('epoch', ei)
        key, subkey = jax.random.split(key)
        reward_pos = jnp.stack(jnp.where(env_state['grid']==2)[1:],axis=1)

        env_state, buffer_state, actions, hippo_hidden, theta, rewards, done, replayed_hippo_theta_output \
            = model_step(env_state, buffer_state, running_encoder_state, running_hippo_state, running_policy_state,
                         subkey, actions, hippo_hidden, theta,
                         args.n_agents, args.bottleneck_size, args.replay_steps, args.height, args.width,
                         args.visual_prob, temperature=0.05)
        
        replayed_hippo_history, replayed_theta_history, output_history = replayed_hippo_theta_output
        hist_hippo.append(replayed_hippo_history.reshape(-1,args.hidden_size))
        # replay_step * n_agents * hidden_size
#### 这里把所有有意义的都加起来
        place = jnp.argmax(output_history[...,:-1],axis=-1)
        
        for n in range(args.n_agents):
            hist_actions[n].append(actions[n])
            hist_pos[n].append(env_state['current_pos'][n])
            if rewards[n]:
                if rewards[n] == 0.5:
                    hist_reward_pos[n].append(jnp.array((reward_pos[n][0],reward_pos[n][1])))
                    reward_hippo[0].append(replayed_hippo_history[:,n,:])
                else:
                    reward_hippo[1].append(replayed_hippo_history[:,n,:])
                hist_replay_place[n].append(place[:,n]) # replay_step * hw
                
            if done[n]:
                start_p = hist_pos[n].pop()
                start_a = hist_actions[n][-1]
                hist_pos[n].append(env_state['goal_pos'][n])

                state_traj = jnp.concatenate((jnp.stack(hist_pos[n],axis=0),jnp.stack(hist_actions[n],axis=0)),axis=1)

                if hist_reward_pos[n]:
                    reward_pos_traj = jnp.stack(hist_reward_pos[n],axis=0)
                if hist_replay_place[n] and hist_reward_pos[n]:
                    print('state and action traj')
                    print(state_traj)
                    plt.title(f'{n}th agent, total_steps:{len(hist_pos[n])-1}')
                    plt.grid() # 生成网格
                    # print(hist_pos[n])
                    plt.plot(state_traj[:,0],state_traj[:,1])

                    print('replay traj')
                    for replay_output in hist_replay_place[n]:
                        plt.plot(replay_output//10, replay_output%10, c='blue', marker='o', markersize=3)
                        print(jnp.stack((replay_output//10, replay_output%10),axis=-1))

                    plt.scatter(reward_pos_traj[:,0],reward_pos_traj[:,1], marker='*', s=100, c='r')
                    # plt.show()
                    plt.cla()
                # hist_traj[n].append((state_traj, reward_pos_traj))
                hist_pos[n] = [start_p]
                hist_reward_pos[n] = []
                hist_replay_place[n] = []
                hist_actions[n] = [start_a]

        if ei%30==29:
###用颜色标一下某个感兴趣变量会更清楚一些，比如用颜色区分初始位置，或者区分两种replay
            plt.subplot(221)
            plt.title('epoch '+str(ei)+' all hippo hidden')
            pca = PCA(n_components=2)
            ar_hist_hippo = np.concatenate(hist_hippo,axis=0)
            ar_hist_hippo = pca.fit_transform(ar_hist_hippo)
            c_idx = np.arange(8).repeat((ar_hist_hippo.shape[0]//args.replay_steps),)
            plt.scatter(ar_hist_hippo[:,0],ar_hist_hippo[:,1],c=c_idx,cmap='viridis')

            plt.subplot(222)
            plt.title('epoch '+str(ei)+' all mid reward hidden')
            mid_reward_hippo = np.concatenate(reward_hippo[0],axis=0)
            mid_reward_hippo = pca.fit_transform(mid_reward_hippo)
            # (replay_steps*n_agent*n_sample) * hidden_size
            c_idx = np.arange(8).repeat((mid_reward_hippo.shape[0]//args.replay_steps),)
            plt.scatter(mid_reward_hippo[:,0],mid_reward_hippo[:,1],c=c_idx,cmap='viridis')

            plt.subplot(223)
            plt.title('epoch '+str(ei)+' goal hidden')
            goal_hippo = np.concatenate(reward_hippo[1],axis=0)
            goal_hippo = pca.fit_transform(goal_hippo)
            # (replay_steps*n_agent*n_sample) * hidden_size
            c_idx = np.arange(8).repeat((goal_hippo.shape[0]//args.replay_steps),)
            plt.scatter(goal_hippo[:,0],goal_hippo[:,1],c=c_idx,cmap='viridis')

            plt.subplot(224)
            plt.title('epoch '+str(ei)+' all reward hidden')
            all_reward_hippo = np.concatenate((mid_reward_hippo, goal_hippo),axis=0)
            all_reward_hippo = pca.fit_transform(all_reward_hippo)
            # (replay_steps*n_agent*n_sample) * hidden_size
            c_idx = np.arange(8).repeat((all_reward_hippo.shape[0]//args.replay_steps),)
            plt.scatter(all_reward_hippo[:,0],all_reward_hippo[:,1],c=c_idx,cmap='viridis')
            cmap = mpl.cm.viridis
            norm = mpl.colors.BoundaryNorm(np.arange(9), cmap.N, extend='neither')
            plt.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap='viridis'),orientation='horizontal')
            plt.show()
            plt.cla()
            

    
### 探索一下降维的意义，比如是否和agent的策略好坏有关，或者和reward奖励变化有关

### hippocampus 吸引子
### hippo_hidden_state UMAP降维

### reward 调成2看一下来回走
## checkpoint 

if __name__ == '__main__':
    args = train.parse_args()

    main(args)
    
    

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

np.set_printoptions(threshold=np.inf)
def model_step(env_state, buffer_state, encoder_state, hippo_state, policy_state,
               key, actions, hippo_hidden, theta,
               n_agents, bottleneck_size, replay_steps, height, width, visual_prob, temperature,
               args):
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
    if args.no_replay:
        replayed_theta = theta
    if args.no_goal_replay:
        replayed_theta = jnp.where(rewards == 1, theta, replayed_theta)
    
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

# def set_pos(grid, pos, value):
#     grid = grid.at[pos[0], pos[1]].set(value)
#     return grid, grid
# def integrate(grid,goal_pos,hist_pos,hist_reward_pos):
#     integrate_fn = partial(set_pos, value=3)
#     last, trajectory = jax.lax.scan(integrate_fn, grid, hist_pos)
#     trajectory = jax.vmap(set_pos, (0, hist_reward_pos, None), 0)(trajectory, hist_reward_pos, 2)

#     for pos in hist_reward_pos:
#         trajectory = trajectory.at[pos[0],pos[1]].set(2)
#     for hpos in hist_pos:
#         # print(hpos)
#         trajectory = trajectory.at[hpos[0],hpos[1]].set(3)
    
#     trajectory = trajectory.at[goal_pos[0],goal_pos[1]].set(4)
#     print(trajectory)
#     return trajectory

def plot_replay(replay_traj, color):
    for replay_output in replay_traj:
        plt.plot(replay_output//10, replay_output%10, c=color, marker='o', markersize=6)

def plot_trajectory(whole_traj:dict):
    agent_th, state_traj, replay_traj, reward_pos_traj = whole_traj.values()
    plt.title(f'{agent_th}th agent, total_steps:{state_traj.shape[0]-1}')
    plt.grid()
    plt.plot(state_traj[:,0],state_traj[:,1])
    plot_replay(replay_traj[:-1], 'blue')
    plot_replay(replay_traj[-1:], 'red')
    plt.scatter(reward_pos_traj[:,0],reward_pos_traj[:,1], marker='*', s=100, c='r')

def plot_heatmap(reward_pos_traj, heatmap):
    # reward_th * replay_step * hw
    for reward_th in range(len(heatmap)):
        for i in range(args.replay_steps):
            plt.subplot(2,4,i+1)
            plt.ylim(10,0)
            plt.imshow(heatmap[reward_th][i].reshape(args.width,args.height).permute(1,0)[:,::-1])
            place_idx = heatmap[reward_th][i].item()
            plt.title(f'argmax:{place_idx//10, place_idx%10}')
        plt.suptitle(f'reward position {reward_pos_traj[reward_th]}')


def display_trajectory(whole_traj:dict, no_goal_replay=False):
    agent_th, state_traj, replay_traj, reward_pos_traj = whole_traj.values()
    print(f'agent {agent_th}')
    print(f'state and action traj, total_step={state_traj.shape[0]-1}')
    print(state_traj)
    for i, replay_output in enumerate(replay_traj):
        print('replay at reward position:'+str(reward_pos_traj[i]))
        if i==len(replay_traj)-1 and no_goal_replay:
            break
        print(jnp.stack((replay_output//10, replay_output%10),axis=-1))


def plot_dimension_reduction_and_replay(ei:int, dimred_replay:dict, args):
    hist_hippo, reward_hippo, total_steps, total_reward, total_replay_distance_scope = dimred_replay.values()
    fig,[[ax0,ax1,ax2],[ax3,ax4,ax5]] = plt.subplots(2,3)
    ax0.set_title('epoch '+str(ei)+' all hippo hidden')
    pca = PCA(n_components=2)
    ar_hist_hippo = np.concatenate(hist_hippo,axis=0)
    
    ar_hist_hippo = pca.fit_transform(ar_hist_hippo)
    c_idx = np.arange(args.replay_steps).reshape(1,-1).repeat(ar_hist_hippo.shape[0]//args.replay_steps,0).reshape((-1,))
    ax0.scatter(ar_hist_hippo[:,0],ar_hist_hippo[:,1],c=c_idx,cmap=args.colormap)

    ax1.set_title('epoch '+str(ei)+' mid reward hidden')
    if reward_hippo[0]:
        mid_reward_hippo = np.concatenate(reward_hippo[0],axis=0)
        # (replay_step*n_agent) * hidden_size
        ### 这里画出来hiddenstate的histogram，没有什么有意义的结果，不是trivial的
        # for i in range(6):
        #     plt.subplot(2,3,i+1)
        #     plt.hist(mid_reward_hippo[args.replay_steps*(i):args.replay_steps*(i+1),:])
        #     plt.title('histogram of mid reward replay')
        # plt.show()
        # plt.cla()
        mid_reward_hippo = pca.fit_transform(mid_reward_hippo)
        # (replay_steps*n_agent*n_sample) * hidden_size
        c_idx = np.arange(args.replay_steps).reshape(1,-1).repeat(mid_reward_hippo.shape[0]//args.replay_steps,0).reshape((-1,))
        ax1.scatter(mid_reward_hippo[:,0],mid_reward_hippo[:,1],c=c_idx,cmap=args.colormap)

    
    ax2.set_title('epoch '+str(ei)+' goal hidden')
    if reward_hippo[1]:
        goal_hippo = np.concatenate(reward_hippo[1],axis=0)
        goal_hippo = pca.fit_transform(goal_hippo)
        # (replay_steps*n_agent*n_sample) * hidden_size
        c_idx = np.arange(args.replay_steps).reshape(1,-1).repeat(goal_hippo.shape[0]//args.replay_steps,0).reshape((-1,))
        ax2.scatter(goal_hippo[:,0],goal_hippo[:,1],c=c_idx,cmap=args.colormap)

    #此处不再总体地画出他们，因为它们已经很不一样了
    # plt.title('epoch '+str(ei)+' all reward hidden')
    # if reward_hippo[0] and reward_hippo[1]:
    #     all_reward_hippo = np.concatenate((mid_reward_hippo, goal_hippo),axis=0)
    #     all_reward_hippo = pca.fit_transform(all_reward_hippo)
    #     # (replay_steps*n_agent*n_sample) * hidden_size
    #     c_idx = np.arange(args.replay_steps).reshape(1,-1).repeat(all_reward_hippo.shape[0]//args.replay_steps,0).reshape((-1,))
    #     plt.scatter(all_reward_hippo[:,0],all_reward_hippo[:,1],c=c_idx,cmap='viridis')
    prefix = 'epoch '+str(ei)+' histogram of total steps'
    if args.no_replay:
        mid_name = ' without replay'
    elif args.no_goal_replay:
        mid_name = ' without goal replay'
    else:
        mid_name = ' with all replay'
    affix = '\n mean:'+str(np.mean(np.array(total_steps)))
    ax3.set_title(prefix+mid_name+affix)
    # if short_mid_reward_hippo:
    #     ar_short = []
    #     for agent_reward_list in short_mid_reward_hippo:
    #         if agent_reward_list:
    #             for mid_hippo in agent_reward_list:
    #                 ar_short.append(mid_hippo)
    #     ar_short = np.concatenate(ar_short,axis=0)
    #     ar_short = pca.fit_transform(ar_short)
    #     c_idx = np.arange(args.replay_steps).reshape(1,-1).repeat(ar_short.shape[0]//args.replay_steps,0).reshape((-1,))
    #     ax3.scatter(ar_short[:,0],ar_short[:,1],c=c_idx,cmap=colormap)
    if total_steps:
        ax3.hist(total_steps)

    ax4.set_title('histogram of total reward'+f'\n mean:{np.mean(np.array(total_reward))}')
    if total_reward:
        ax4.hist(total_reward)

    ax5.set_title('total replay distance and scope')
    if total_replay_distance_scope[0] and total_replay_distance_scope[1]:
        ax5.bar(['distance','scope'],[np.mean(np.array(total_replay_distance_scope[0])),np.mean(np.array(total_replay_distance_scope[1]))])

    cmap = mpl.cm.get_cmap(args.colormap)
    new_cmap = mpl.colors.ListedColormap([cmap(i) for i in np.linspace(0, 1, args.replay_steps)])
    # norm = mpl.colors.BoundaryNorm(np.arange(args.replay_steps+1), cmap.N, extend='neither')
    norm = mpl.colors.Normalize(vmin=0, vmax=args.replay_steps)
    fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=new_cmap))


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
    hist_replay_place_map = [[] for _ in range(args.n_agents)]
    hist_actions = [[] for _ in range(args.n_agents)]
    hist_hippo = []
    reward_hippo = [[],[]]
    total_steps = []
    total_reward = []
    total_replay_distance_scope = [[],[]]
    reward_pos_traj = None
    comparison_pts = jnp.zeros(args.n_agents,dtype=jnp.int32)
    # short_mid_reward_hippo = [[] for _ in range(args.n_agents)]
    ### 短路径的mid-reward-replay和所有的mid-reward-replay没有明显区别
    # 0 for mid-path reward replay and 1 for goal replay

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
                         args.visual_prob, temperature=0.05,
                         args=args)
        
        replayed_hippo_history, replayed_theta_history, output_history = replayed_hippo_theta_output
        # replay_hippo_theta_output: (replay_steps, n_agents, hidden_size), (replay_steps, n_agents, hidden_size)
        hist_hippo.append(replayed_hippo_history.reshape(-1,args.hidden_size))
        # replay_step * n_agents * hidden_size
        place_map = output_history[...,:-1]
        # replay_step * n_agents * hw
        max_decoding_place = jnp.argmax(output_history[...,:-1],axis=-1)
        # replay_step * n_agents
        
        for n in range(args.n_agents):
            hist_actions[n].append(actions[n])
            hist_pos[n].append(env_state['current_pos'][n])
            if rewards[n]:
                # including mid_reward and goal
                if rewards[n] != 1:
                    hist_reward_pos[n].append(jnp.array((reward_pos[n][0],reward_pos[n][1])))
                    reward_hippo[0].append(replayed_hippo_history[:,n,:])
                else:
                    hist_reward_pos[n].append(jnp.array((env_state['goal_pos'][n])))
                    reward_hippo[1].append(replayed_hippo_history[:,n,:])
                hist_replay_place[n].append(max_decoding_place[:,n]) # replay_step * hw
                hist_replay_place_map[n].append(place_map[:,n,:])
                total_reward.append(rewards[n])
                
            if done[n]:
                start_p = hist_pos[n].pop()
                start_a = hist_actions[n][-1]
                hist_pos[n].append(env_state['goal_pos'][n])
                total_steps.append(len(hist_pos[n])-1)

                interest_condition = (not args.only_reward_trajectory \
                    or (args.only_reward_trajectory and len(hist_reward_pos[n])>1))
                if interest_condition:
                    state_traj = jnp.concatenate((jnp.stack(hist_pos[n],axis=0),jnp.stack(hist_actions[n],axis=0)),axis=1)
                    reward_pos_traj = jnp.stack(hist_reward_pos[n],axis=0)

                    whole_traj = {'agent_th':n, 'state_traj':state_traj, 'replay_traj':hist_replay_place[n], 'reward_pos_traj':reward_pos_traj}
                    hist_traj[n].append(whole_traj)
                    if args.output_traj:
                        display_trajectory(whole_traj, args.no_goal_replay)
                        plot_trajectory(whole_traj)
                        plt.show()
                        plt.cla()
                        if args.output_heatmap:
                            plot_heatmap(reward_pos_traj, hist_replay_place_map[n])
                            plt.show()
                            plt.cla()
                    for i, replay_output in enumerate(hist_replay_place[n]):
                        if i==len(hist_replay_place[n])-1 and args.no_goal_replay:
                            break
                        replay_distance = jnp.sqrt(jnp.square(jnp.diff(replay_output//10))+jnp.square(jnp.diff(replay_output%10))).sum()
                        replay_scope = (jnp.max(replay_output//10)-jnp.min(replay_output//10)) \
                                        + (jnp.max(replay_output%10)-jnp.min(replay_output%10))
                        total_replay_distance_scope[0].append(replay_distance)
                        total_replay_distance_scope[1].append(replay_scope)
                hist_pos[n] = [start_p]
                hist_reward_pos[n] = []
                hist_replay_place[n] = []
                hist_replay_place_map[n] = []
                hist_actions[n] = [start_a]
                reward_pos_traj = None

        if args.output_dimension_reduction and ei%args.epochs_per_output==args.epochs_per_output-1:
            dimred_replay = {'hist_hippo':hist_hippo, 'reward_hippo':reward_hippo, 'total_steps':total_steps,\
                 'total_reward':total_reward, 'total_replay_distance_scope':total_replay_distance_scope}
            plot_dimension_reduction_and_replay(ei, dimred_replay, args)
            plt.show()
            plt.cla()
        
        if args.output_comparison:
            for n in range(args.n_agents):
                if len(hist_traj[n])>=args.pics_per_output+comparison_pts[n]:
                    for i in range(args.pics_per_output):
                        plt.subplot(2,args.pics_per_output//2,i+1)
                        print(len(hist_traj[n]))
                        print(comparison_pts[n])
                        whole_traj = hist_traj[n][i+comparison_pts[n]]
                        display_trajectory(whole_traj)
                        plot_trajectory(whole_traj)
                    plt.show()
                    plt.cla()
                    comparison_pts = comparison_pts.at[n].add(args.pics_per_output)
                    print(comparison_pts[n])

            
# （已完成，使用heatmap和argmax直接解码基本没什么区别，heatmap图中都是高斯的）把hippo_output的heatmap画出来，而不是只有argmax
# hippo_hidden_state UMAP降维
# （已完成：replay降维结果和agent走到终点的策略好坏没关系，很快走到终点的agent的流形和所有的agent的流形一样）探索一下降维的意义，比如是否和agent的策略好坏有关
# （已完成：仅仅阻断goal replay之后探索速度减慢并且平均奖励略有减少）阻断一下goal replay， 探索一下它为什么是6个

# （ing）mid-reward replay目前的发现：降维结果分成两坨，一坨是奇数步，另一坨是偶数步……还不清楚为什么
### 进一步的发现：在huge_reward中mid结构更加明显，是一步一步的扩散行为，goal结构更乱
### 但在small_reward中，mid分成两坨（无论是replay_steps==4 or 8），goal明显只有六个部分

# （已完成，当reward很大的时候可以看出明显结构）看一下hippo具体的流形怎么样，具体来说是每一个step都画一张图
# 按照位置标颜色，而不是按照具体是replay的哪一步
# （已完成但存疑，可以通过调用output_comparison来测试，agent不太会根据上一次奖励在哪里来规划下一步动作……不知道replay的意义是什么）记一下在奖励位置先后遇到两次奖励的replay，看看位置变化会不会引起replay变化
### 上边这个重点是关注同一个agent在两次episode之间的策略变化，后边重点关注一下这个
# （已完成，当replay步数减少到4的时候，模型在实际的过程中只能replay两步，基本形状和replay_steps=8一样，但reward改变会使得replay更加激进，结构更加明显，可以通过改变replay_steps来测试）replay步数改一下
# （已完成，阻断所有的replay之后agent会一直撞墙，说明replay有明显加速效果）测试时候阻断一下replay
# hidden_output
# theta降一下维


### hippocampus 吸引子

# （已完成，在huge_reward的训练情况下，有一些agent学会了来回走来获取原地奖励，但大多数agent的replay策略更加激进，走得更快，不知道为什么，感觉reward随机改变的概率还是太高？）reward 调成2看一下来回走
## checkpoint 
### 障碍物

if __name__ == '__main__':
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
    parser.add_argument('--load_encoder', type=str, default='./modelzoo/r_input_encoder/r_input_encoder_reward2')  # todo: checkpoint
    parser.add_argument('--load_hippo', type=str, default='./modelzoo/r_input_hippo/r_input_hippo_reward2')
    parser.add_argument('--load_policy', type=str, default='./modelzoo/r_policy_reward2_hippo_retrain')
    parser.add_argument('--hidden_size', type=int, default=128)

    # visualization
    parser.add_argument('--colormap', type=str, default='Set1')
    parser.add_argument('--output_traj', '-a', action='store_true', default=False)
    parser.add_argument('--only_reward_trajectory', '-r', action='store_true' ,default=False)
    parser.add_argument('--output_dimension_reduction', '-d', action='store_true', default=False)
    parser.add_argument('--epochs_per_output', type=int, default=30,
                        help='how many epochs before output of dimension reduction')
    parser.add_argument('--no_replay', action='store_true', default=False)
    parser.add_argument('--no_goal_replay', action='store_true', default=False)
    parser.add_argument('--output_heatmap', action='store_true', default=False)
    parser.add_argument('--output_comparison', '-c', action='store_true', default=False,
                        help='whether to output continuous trajectories of one agent')
    parser.add_argument('--pics_per_output',type=int, default=4,
                        help='how many trajectories to be showed of one agent in output')


    args = parser.parse_args()
    main(args)
    
    
### 不要定义成policy network，因为既有memory consolidation也有planning
### hippo 和 encoder 统一在一起
### 不要分步训练，看看replay会不会有什么变化


### 改一下reward的平均方法，让它对每一次episode做平均
### 改一下hippo的position coding...
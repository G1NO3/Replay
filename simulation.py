import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
# from sklearn.decomposition import PCA
import matplotlib as mpl
# hist_pos = [[1,2],[2,1],[0,0]]
# b = jnp.zeros((3,3))
# # for pos in a:
#     # b = b.at[(pos[0],pos[1])].set(3)
# a = jnp.eye(3)
# rewards = jnp.array([1,1,0])
# subkey = jax.random.PRNGKey(0)
# reset_flag = jax.random.uniform(subkey, (rewards.shape[0], 1, 1)) < 0.8
# # print(jnp.where(a))
# grid = jnp.ones((5,5))
# new_grid = jnp.where((rewards.reshape((-1, 1, 1)) > 0) & reset_flag, 0, grid)
# # print(new_grid)
# plt.grid()
# plt.plot(np.arange(3),np.arange(3))
# # plt.show()
# def f(z,xs):
#     x, y = z
#     return (x*y, x+y), (x*y, x+y, jnp.dot(x,y))
# end, result = jax.lax.scan(f, init=(jnp.arange(2)*1.,jnp.ones(2)*1.), xs=None, length=3)
# A = jnp.array([[[1,4],[2,3]]])
# print(jnp.argmax(A,axis=-1))
# print(A.reshape((2,2)))
# fig = plt.figure()
# cmap = mpl.cm.viridis
# print(cmap.N)
# # norm = mpl.colors.CenteredNorm(4,4)
# norm = mpl.colors.BoundaryNorm(np.arange(9), cmap.N, extend='neither')
# fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap='viridis'))
# a = np.random.normal(0,0.5,(300,)).reshape((-1,3))
# b = np.random.normal(5,0.5,(300,)).reshape((-1,3))
# c = np.concatenate((a,b),axis=0)
# pca = PCA(n_components=2)
# d = pca.fit_transform(a)
# e = pca.fit_transform(b)
# f = pca.fit_transform(c)
# print(f.shape)
# print(d.shape)
# plt.scatter(d[:,0],d[:,1],c='r')
# plt.scatter(e[:,0],e[:,1],c='b')
# plt.scatter(f[:,0],f[:,1],c='g')
# n = np.arange(6).reshape(1,-1).repeat(6,axis=0).reshape((-1,))
# print(n)
# plt.show()

# fig,[ax0,ax1] = plt.subplots(2,1)
# c_idx = np.arange(4)
# x = c_idx
# # axis.scatter(x,x,c=c_idx,cmap='Set1')
# colors = ['red','yellow','blue','green']
# newcmap = mpl.colors.ListedColormap(colors)

# norm = mpl.colors.Normalize(vmin=0, vmax=4)
# fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=newcmap))
# plt.show()

# batch = {'obs_embed': batch[0], 'action_embed': batch[1], 'his_hippo_hidden': batch[2], 'his_theta': batch[3],
#             'his_rewards': batch[4], 'his_action': batch[5], 'his_logits': batch[6], 'his_values': batch[7]
#             }
# for k in batch:
#     print(k)
#     print(batch[k])
#     if batch[k] == 'his_rewards':  # fixme: his_rewards is of t-1, align it with other
#         batch[k] = batch[k][1:]
#     else:
#         batch[k] = batch[k][:-1]
# print(batch)

# key = jax.random.PRNGKey(0)
# key, subkey = jax.random.split(key)
# # a = jax.random.randint(key, (2,3,4),0,3)
# # b = jax.random.randint(subkey, (2,3,4),0,3)
# a = jnp.ones((2,3,4))
# b = jnp.ones((2,3,4))*2
# def addone(xy,xs):
#     x, y = xy
#     return (x+1,y+2),(x+1,y+2)
# (_a,_b), c = jax.lax.scan(addone,init=(a,b),xs=None,length=8)
# print(c[0].shape)

# replay_fn_to_scan = partial(replay_fn_for_seq, policy_params=policy_params,
#                             hippo_state=hippo_state, policy_state=policy_state,
#                             n_agents=n_agents,
#                             obs_embed_size=batch['obs_embed'].shape[-1],
#                             action_embed_size=batch['action_embed'].shape[-1])
# (_, his_replayed_theta), _ = jax.lax.scan(replay_fn_to_scan,
#                                             init=(batch['his_hippo_hidden'], batch['his_theta']),
#                                             xs=None, length=replay_steps)


# @partial(jax.jit, static_argnums=(5, 6, 7))
# def replay_fn_for_seq(hippo_and_theta, xs, policy_params, hippo_state, policy_state,
#                       n_agents, obs_embed_size, action_embed_size):
#     # the only difference with replay_fn is that this func replays for whole seq [t, ..., ...]
#     # to match the input/output stream of jax.lax.scan
#     # and also need to calculate grad of policy_params
#     hippo_hidden, theta = hippo_and_theta
#     new_theta, (policy, value, to_hipp) = policy_state.apply_fn({'params': policy_params},
#                                                                 theta, jnp.zeros(
#             (hippo_hidden.shape[0], n_agents, obs_embed_size)),
#                                                                 hippo_hidden)
#     new_hippo_hidden, _ = hippo_state.apply_fn({'params': hippo_state.params},
#                                                hippo_hidden, to_hipp,
#                                                (jnp.zeros((hippo_hidden.shape[0], n_agents, obs_embed_size)),
#                                                 jnp.zeros((hippo_hidden.shape[0], n_agents, action_embed_size))),
#                                                jnp.zeros((hippo_hidden.shape[0], n_agents, 1)))
#     return (new_hippo_hidden, new_theta), (new_hippo_hidden, new_theta)

a = 0.
b = jnp.sqrt((((a+5)/2-2.5+1.2)**2-1.4))-0.2
print(b)
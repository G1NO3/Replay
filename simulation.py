import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
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
a = jnp.zeros(4,dtype=jnp.int32)
for i in range(3):
    a.at[i].add(i)
print(a)
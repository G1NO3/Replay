import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
hist_pos = [[1,2],[2,1],[0,0]]
b = jnp.zeros((3,3))
# for pos in a:
    # b = b.at[(pos[0],pos[1])].set(3)
a = jnp.eye(3)
rewards = jnp.array([1,1,0])
subkey = jax.random.PRNGKey(0)
reset_flag = jax.random.uniform(subkey, (rewards.shape[0], 1, 1)) < 0.8
# print(jnp.where(a))
grid = jnp.ones((5,5))
new_grid = jnp.where((rewards.reshape((-1, 1, 1)) > 0) & reset_flag, 0, grid)
# print(new_grid)
plt.grid()
plt.plot(np.arange(3),np.arange(3))
# plt.show()
def f(z,xs):
    x, y = z
    return (x*y, x+y), (x*y, x+y, jnp.dot(x,y))
end, result = jax.lax.scan(f, init=(jnp.arange(2)*1.,jnp.ones(2)*1.), xs=None, length=3)
A = jnp.array([[[1,4],[2,3]]])
print(jnp.argmax(A,axis=-1))
print(A.reshape((2,2)))
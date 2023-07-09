"""
Env, a 10*10 grid env
obs: (10, 10), 0 for pathway, 1 for obstacle, 2 for reward, 3 for self pos,
    (the obs is set to be zero with 0.95 prob in tasks now)
goal: fixed at (9, 9), when reached the goal, the agent will be send back to the start point
reward: a small reward random appear in the map. When got the reward,
    the reward will be send to another pos with 0.1 prob (call reset_reward)
action: 4 actions, up, down, left, right
"""
import jax
import jax.numpy as jnp
from functools import partial


# fixme: seldom obs, whether reset randomly,

@partial(jax.jit, static_argnums=(2,))
def add_obstacle(grid, obstacles, n_agents):
    # add obstacle for grid [n, h, w];
    # obstacles: n * n_obstacles * [2,]
    for na in range(n_agents):
        for obst in obstacles[na]:
            grid = grid.at[obst].set(1)
    return grid


@partial(jax.jit, static_argnums=(2, 3, 4))
def add_reward(grid, key, n_agents, height, width):
    key, *subkeys = jax.random.split(key, 3)
    reward_x = jax.random.randint(subkeys[0], (n_agents,1), minval=1, maxval=height - 1)  # fixme: h-1/h-2?
    reward_y = jax.random.randint(subkeys[1], (n_agents,1), minval=1, maxval=width - 1)

    # fixme: one reward for each env
    def set_r(gd, pos):
        # gd[10, 10], pos[2,]
        return gd.at[pos[0], pos[1]].set(2)

    grid = jax.vmap(set_r, (0, 0), 0)(grid, jnp.stack((reward_x, reward_y), axis=1))  # todo
    grid = jax.vmap(set_r, (0, 0), 0)(grid, jnp.stack((reward_x + 1, reward_y), axis=1))
    grid = jax.vmap(set_r, (0, 0), 0)(grid, jnp.stack((reward_x - 1, reward_y), axis=1))
    grid = jax.vmap(set_r, (0, 0), 0)(grid, jnp.stack((reward_x, reward_y + 1), axis=1))
    grid = jax.vmap(set_r, (0, 0), 0)(grid, jnp.stack((reward_x, reward_y - 1), axis=1))
    reward_center = jnp.concatenate((reward_x, reward_y),axis=1)
    return grid, reward_center


@jax.jit
def fetch_pos(grid, pos):
    # grid[h, w], pos[2,]
    return grid[pos[0], pos[1]]


@jax.jit
def set_pos(grid, pos, value):
    # grid[h, w], pos[2,]
    return grid.at[pos[0], pos[1]].set(value)


@jax.jit
def prepare_obs(grid, current_pos):
    # obs: can see self.current pos, but cannot see rewards
    obs = jax.vmap(set_pos, (0, 0, None), 0)(grid, current_pos, 3)
    obs = jnp.where(obs == 2, 0, obs)
    return obs


@jax.jit
def take_action(actions, current_pos, grid, goal_pos):
    next_pos = jnp.where(actions == 0, current_pos - jnp.array([1, 0]),
                         jnp.where(actions == 1, current_pos + jnp.array([0, 1]),
                                   jnp.where(actions == 2, current_pos + jnp.array([1, 0]),
                                             jnp.where(actions == 3, current_pos - jnp.array([0, 1]),
                                                       current_pos))))
    # next_pos [n, 2]
    next_pos = jnp.clip(next_pos, 0, jnp.array([grid.shape[1] - 1, grid.shape[2] - 1]))

    hit = jax.vmap(fetch_pos, (0, 0), 0)(grid, next_pos)
    hit = hit.reshape((-1, 1))
    blocked = jnp.where(hit == 1, -1, 0)
    rewarded = jnp.where(hit == 2, 0.5, 0)
    # step_punishment = -jnp.ones((actions.shape[0],1))*0.03

    rewards = jnp.where(jnp.all(next_pos == goal_pos, axis=1, keepdims=True), 1, 0) + blocked + rewarded\
                # + step_punishment
    done = jnp.all(next_pos == goal_pos, axis=1)
    return next_pos, rewards, done, blocked


def reset(width, height, n_agents, key):
    grid = jnp.zeros((n_agents, height, width), dtype=jnp.int8)
    grid = add_obstacle(grid, [[] for _ in range(n_agents)], n_agents)
    # fixme: no obstacles now; so add_obstacle is not checked
    # fixme: magic number: 0 for pathway, 1 for obstacle, 2 for reward, 3 for self pos
    grid, reward_center = add_reward(grid, key, n_agents, height, width)
    start_pos = jnp.array([[0, 0]] * n_agents)
    goal_pos = jnp.array([[height - 1, width - 1]] * n_agents)
    current_pos = start_pos

    return prepare_obs(grid, current_pos), {'grid': grid, 'current_pos': current_pos,
                                            'goal_pos': goal_pos, 'reward_center':reward_center}


@jax.jit
def step(env_state, actions):
    next_pos, rewards, done, blocked = take_action(actions,
                                                   env_state['current_pos'], env_state['grid'],
                                                   env_state['goal_pos'])
    current_pos = jnp.where(blocked == -1, env_state['current_pos'], next_pos)
    current_pos = jnp.where(done.reshape(-1, 1), jnp.zeros_like(current_pos), current_pos)
    # fixme: reset pos to zero(start point) as soon as goal is reached

    obs = prepare_obs(env_state['grid'], current_pos)
    env_state = dict(env_state, current_pos=current_pos)

    return obs, rewards, done, env_state


@jax.jit
def reset_reward(env_state, rewards, key):
    key, subkey = jax.random.split(key)
    reset_flag = jax.random.uniform(subkey, (rewards.shape[0], 1, 1)) < 0.1
    new_grid = jnp.where((rewards.reshape((-1, 1, 1)) > 0) & reset_flag, 0, env_state['grid'])
    # fixme: if reward, set grid to 0 (no obstacles)
    new_grid, new_center = add_reward(new_grid, key, *env_state['grid'].shape)
    new_grid = jnp.where((rewards.reshape((-1, 1, 1)) > 0) & reset_flag, new_grid, env_state['grid'])
    new_center = jnp.where((rewards.reshape((-1, 1)) > 0) & reset_flag.reshape(-1,1), new_center, env_state['reward_center'])
    env_state = dict(env_state, grid=new_grid, reward_center=new_center)

    return env_state


# @jax.jit
# def get_reawrd_pos(env_state):
#     reward_pos = jnp.stack(jnp.where(env_state['grid'] == 2), axis=0)
#     return reward_pos


if __name__ == '__main__':
    obs, env_state = reset(8, 8, 3, jax.random.PRNGKey(0))
    print(obs, )
    actions = jnp.ones((3, 1), dtype=jnp.int8) * 2
    obs, rewards, done, env_state = step(env_state, actions)
    print(obs, rewards, done, )

"""
Config for path_int.py
"""

# env
width = 10
height = 10
n_agents = 1024  # batch_size
sigma = 1.  # the std of place cell
visual_prob = 0.05  # the prob that obs is not masked to zero
mid_reward = 2.  # the reward at the middle
# network
hidden_size = 128  # hidden_size of hippo campus state
bottleneck_size = 8  # the dim of pfc's input to hippo

# optimizer
lr = 5e-4
wd = 1e-3
epoch = int(1e6)
train_every = 8
sample_len = 256

# buffer
max_size = 2000

# other
save_name = 'r_input'
load = 'modelzoo/r_input_encoder/checkpoint_40000'

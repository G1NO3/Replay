"""
Config for path_int.py
"""

# env
width = 8
height = 8
n_agents = 2048  # batch_size
sigma = 2.  # the std of place cell
visual_prob = 1.  # the prob that obs is not masked to zero

# network
hidden_size = 128  # hidden_size of hippo campus state
bottleneck_size = 8  # the dim of pfc's input to hippo

# optimizer
lr = 1e-4
wd = 1e-3
epoch = int(3e6)
train_every = 16
sample_len = 256

# buffer
max_size = 20000

# other
save_name = 'env8'
load = 'modelzoo/env8_encoder/checkpoint_385000'

import jax
import jax.numpy as jnp
from flax import linen as nn  # Linen API


class Encoder(nn.Module):
    """Encoder module that embeds obs and a_t-1"""

    @nn.compact
    def __call__(self, obs, action):
        # obs [n, h, w], action [n, 1]
        obs = nn.Embed(4, 4)(obs)  # [n, h, w, 4]
        # obs = jnp.transpose(obs, axes=[0, 3, 1, 2])  # fixme: bug here
        x = nn.Conv(features=8, kernel_size=(3, 3), strides=2)(obs)
        x = nn.tanh(x)
        x = nn.Conv(features=16, kernel_size=(3, 3), strides=2)(x)
        x = nn.tanh(x)
        x = x.reshape((x.shape[0], -1))  # flatten

        xa = nn.Embed(4, 4)(action)  # [n, 1, 4]
        xa = nn.tanh(xa)
        xa = xa.reshape((xa.shape[0], -1))  # [n, 4]

        return x, xa


class Policy(nn.Module):
    """A simple pfc model."""
    output_size: int
    hidden_size: int
    bottleneck_size: int

    @nn.compact
    def __call__(self, theta, obs_embeds, hipp_hidden):
        # obs_embeds [n, d](-1~1), theta[n, h](-1~1), hipp_hidden[n, h] (often zero, (-1~1))
        # return new_theta[n, h](-1~1), output(policy logits)[n, output_size], value[n, 1], to_hipp[n, output_size]
        # during walk, theta is fixed; during replay, theta is updated
        hipp_info = nn.tanh(nn.Dense(self.bottleneck_size)(hipp_hidden))  # [n, bottleneck_size]
        new_theta = nn.tanh(nn.Dense(self.hidden_size)(jnp.concatenate((obs_embeds, theta, hipp_info), axis=-1)))
        output = nn.Dense(self.output_size)(new_theta)
        value = nn.Dense(1)(new_theta)  # fixme: separate actor and critic networks can achieve better results
        to_hipp = nn.tanh(nn.Dense(self.bottleneck_size)(new_theta))
        # new_theta = jnp.where(hipp_hidden.sum(axis=-1).reshape(-1, 1) > 0, new_theta, theta)  # fixme:
        return new_theta, (output, value, to_hipp)


class Hippo(nn.Module):
    """A simple hippo model.(rnn)"""
    hidden_size: int
    output_size: int

    @nn.compact
    def __call__(self, hipp_hidden, pfc_input, encoder_inputs, rewards):
        # pfc_input [n, d](-1~1), encoder_input(obs_embed[n, d], action_embed[n, d])(-1~1)), hipp_hidden[n, d](-1~1)
        # rewards[n, 1]
        obs_embed, action_embed = encoder_inputs
        new_hidden = nn.Dense(features=self.hidden_size)(jnp.concatenate(
            (obs_embed, action_embed, pfc_input, hipp_hidden, rewards), axis=-1))
        new_hidden = nn.tanh(new_hidden)
        output = nn.Dense(self.output_size)(new_hidden)
        return new_hidden, output


if __name__ == '__main__':
    pass

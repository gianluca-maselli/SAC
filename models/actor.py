import torch
import torch.nn as nn
from torch.distributions import Normal
from SAC.common import weights_init_, LinearLayer
import math
import torch.nn.functional as F

class Policy(nn.Module):
    def __init__(self, input_size, num_actions, hidden_size, mode, action_space=None):
        super(Policy, self).__init__()

        self.mode = mode
        self.LOG_SIG_MIN = -20
        self.LOG_SIG_MAX = 2
        self.epsilon = 1e-6

        self.MLP = nn.Sequential(
                LinearLayer(input_size, hidden_size),
                LinearLayer(hidden_size, hidden_size),
                )
        self.mean_linear = nn.Linear(hidden_size, num_actions)
        #case of gaussian policy
        self.log_std_linear = nn.Linear(hidden_size, num_actions)
        #case of deterministic policy
        self.noise = torch.Tensor(num_actions)

        self.apply(weights_init_)
        # action rescaling used in DMC control 
        self.action_scale = torch.tensor(1.)
        self.action_bias = torch.tensor(0.)
        
    def forward(self, state):
        x = self.MLP(state)
        mean_ = self.mean_linear(x)

        if self.mode == 'Gaussian':
            log_std = self.log_std_linear(x)
            log_std = torch.clamp(log_std, min=self.LOG_SIG_MIN, max=self.LOG_SIG_MAX)
            return mean_, log_std
        else:
            mean = torch.tanh(mean_) * self.action_scale + self.action_bias
            return mean

    def sample(self, state):
        """
        Gaussian policy:
            - The policy outputs the mean and log standard deviation of a Gaussian distribution for each action dimension.
            - The action is sampled from this Gaussian distribution by using the reparameterization trick.
            - The sampled action is passed through a tanh to ensure it stays within bounds.
            - Encourages exploration due to its inherent randomness.
        """
        if self.mode == 'Gaussian':
            # get the mean and log standard deviation of the Gaussian distribution from the policy network
            mean, log_std = self.forward(state)
            ## suggested by Ilya for stability (it is not used in normal SAC)
            log_std = torch.tanh(log_std)
            log_std = self.LOG_SIG_MIN + 0.5 * (self.LOG_SIG_MAX - self.LOG_SIG_MIN) * (log_std+1)
            # convert log_std to standard deviation
            std = log_std.exp()
            # create a Gaussian distribution using the mean and std
            normal = Normal(mean, std)
            # sample from the distribution using the reparameterization trick:
            # x_t = mean + std * noise, where noise ~ N(0, 1)
            x_t = normal.rsample()
            # apply tanh to have the action into the range (-1, 1)
            y_t = torch.tanh(x_t)
            # rescale the action to the environment action range
            action = y_t * self.action_scale + self.action_bias
            #calculate log probability of the sampled action before the tanh
            log_prob = normal.log_prob(x_t)
            # adjust log probability to account for the tanh transformation
            # using the change of variables formula for probability densities
            # Squash correction (from original SAC implementation)
            #log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + self.epsilon)
            # improved squash correction
            log_prob = log_prob - (2. * (math.log(2.) - x_t - F.softplus(-2. * x_t)))
            # sum the log probabilities across action dimensions
            log_prob = log_prob.sum(1, keepdim=True)
            # compute the mean action after squashing and rescaling
            mean = torch.tanh(mean) * self.action_scale + self.action_bias
            return action, log_prob, mean
        else:
            """
            Deterministic Policy:
                - The policy directly outputs a single action (no distribution).
                - No sampling is involved during action selection.
                - Noise is added externally.
            """
            # get the action output from the deterministic policy network.
            mean = self.forward(state)
            # generate exploration noise sampled from a normal distribution
            noise = self.noise.normal_(0., std=0.1)
            # limit the magnitude of noise to prevent excessive exploration.
            noise = noise.clamp(-0.25, 0.25)
            # add the noise to the action to increase exploration
            action = mean + noise
            action = torch.clamp(action, -1.0, 1.0)
            # the action is deterministic, there no log-probability, it is set to 0 as a placeholder.
            return action, torch.tensor(0.), mean

    def to(self, device):
        self.action_scale = self.action_scale.to(device)
        self.action_bias = self.action_bias.to(device)
        if self.mode != 'Gaussian':
            self.noise = self.noise.to(device)
        return super(Policy, self).to(device)

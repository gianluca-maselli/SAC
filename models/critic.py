from SAC.common import weights_init_, LinearLayer
import torch
import torch.nn as nn

"""
  The algorithm makes use of two soft Q-functions to mitigate positive bias in
  the policy improvement step that is known to degrade performance of value based methods.
"""
class QNet(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(QNet, self).__init__()
        #Q1 Network
        self.Q1 = nn.Sequential(
                LinearLayer(input_size, hidden_size),
                LinearLayer(hidden_size, hidden_size),
                nn.Linear(hidden_size, 1)
                )
        #Q2 Network
        self.Q2 = nn.Sequential(
                LinearLayer(input_size, hidden_size),
                LinearLayer(hidden_size, hidden_size),
                nn.Linear(hidden_size, 1)
                )

        self.apply(weights_init_)

    def forward(self, state, action):
        assert state.size(0) == action.size(0)
        x = torch.cat([state, action], 1)
        q1 = self.Q1(x)
        q2 = self.Q2(x)
        return q1, q2

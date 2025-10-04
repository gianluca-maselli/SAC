import torch 
import numpy as np

class ReplayBuffer:
    def __init__(self, obs_shape, action_shape, buffer_size, batch_size, device):
        self.buffer_size = buffer_size
        self.device = device
        self.pos = 0
        self.full = False
        self.batch_size = batch_size

        self.observations = np.empty((buffer_size, *obs_shape), dtype=np.float32)
        self.next_observations = np.empty((buffer_size, *obs_shape), dtype=np.float32)
        self.actions = np.empty((buffer_size, *action_shape), dtype=np.float32)
        self.rewards = np.empty((buffer_size, 1), dtype=np.float32)
        self.dones = np.empty((buffer_size, 1), dtype=np.float32)

    def add(self, obs, next_obs, action, reward, done):
        # Copy to avoid modification by reference
        np.copyto(self.observations[self.pos], obs)
        np.copyto(self.actions[self.pos], action)
        np.copyto(self.rewards[self.pos], reward)
        np.copyto(self.next_observations[self.pos], next_obs)
        np.copyto(self.dones[self.pos], done)

        
        self.pos = (self.pos + 1) % self.buffer_size
        if self.pos == self.buffer_size:
            self.full = True
            self.pos = 0
    
    def __len__(self,):
        return self.buffer_size if self.full else self.pos
    
    def sample(self,):
        idxs = np.random.randint(0, self.buffer_size if self.full else self.pos, size=self.batch_size)

        observations = torch.as_tensor(self.observations[idxs], device=self.device).float()
        next_observations = torch.as_tensor(self.next_observations[idxs], device=self.device).float()
        actions = torch.as_tensor(self.actions[idxs], device=self.device).float()
        rewards = torch.as_tensor(self.rewards[idxs], device=self.device).float()
        dones = torch.as_tensor(self.dones[idxs], device=self.device).float()
        return observations, actions, rewards, next_observations, dones

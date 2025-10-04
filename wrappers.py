#wrappers: https://github.com/evgenii-nikishin/rl_with_resets/tree/main/continuous_control/wrappers

import copy
from typing import Dict, Optional, OrderedDict
import numpy as np
from dm_control import suite
from dm_env import specs
#from gym import core, spaces
from typing import Tuple
#import gym
#from gym.wrappers import RescaleAction
#from gym.spaces import Box, Dict
import gymnasium as gym
from gymnasium import core, spaces
from gymnasium.wrappers import RescaleAction
from gymnasium.spaces import Box, Dict
import imageio
import sys
import time
import os 

#TimeStep = Tuple[np.ndarray, float, bool, dict]
TimeStep = Tuple[np.ndarray, float, bool, bool, dict]

def dmc_spec2gym_space(spec):
    if isinstance(spec, OrderedDict):
        spec = copy.copy(spec)
        for k, v in spec.items():
            spec[k] = dmc_spec2gym_space(v)
        return spaces.Dict(spec)
    elif isinstance(spec, specs.BoundedArray):
        return spaces.Box(low=spec.minimum,
                          high=spec.maximum,
                          shape=spec.shape,
                          dtype=spec.dtype)
    elif isinstance(spec, specs.Array):
        return spaces.Box(low=-float('inf'),
                          high=float('inf'),
                          shape=spec.shape,
                          dtype=spec.dtype)
    else:
        raise NotImplementedError

class DMCEnv(core.Env):
    def __init__(self, domain_name: str, task_name: str, task_kwargs: Optional[Dict] = {}, environment_kwargs=None):
        assert 'random' in task_kwargs, 'please specify a seed, for deterministic behaviour'

        self._env = suite.load(domain_name=domain_name,
                               task_name=task_name,
                               task_kwargs=task_kwargs,
                               environment_kwargs=environment_kwargs)
        self.action_space = dmc_spec2gym_space(self._env.action_spec())

        self.observation_space = dmc_spec2gym_space(self._env.observation_spec())

        #self.seed(seed=task_kwargs['random'])
        #self._env.reset(seed=task_kwargs['random'])

    def __getattr__(self, name):
        return getattr(self._env, name)

    
    def step(self, action: np.ndarray) -> TimeStep:
        assert self.action_space.contains(action)

        time_step = self._env.step(action)
        reward = time_step.reward or 0
        obs = time_step.observation
        info = {}
        
        if time_step.last():
            if time_step.discount == 1.0:
                terminated = False   # not a natural terminal
                truncated = True     # hit the time limit
            else:
                terminated = True    # natural termination (fell, etc.)
                truncated = False
        else:
            terminated = False
            truncated = False

        return obs, reward, terminated, truncated, info
    
    #return time_step.observation
    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        # Optionally handle seed if needed:
        if seed is not None:
            # dm_control's suite.load doesn't support re-seeding directly after init
            pass
        time_step = self._env.reset()
        info = {}
        return time_step.observation, info

    def render(self, mode='rgb_array', height: int = 84, width: int = 84, camera_id: int = 0):
        assert mode == 'rgb_array', 'only support rgb_array mode, given %s' % mode
        return self._env.physics.render(height=height, width=width, camera_id=camera_id)


class SinglePrecision(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)

        if isinstance(self.observation_space, Box):
            obs_space = self.observation_space
            self.observation_space = Box(obs_space.low, obs_space.high,
                                         obs_space.shape)
        elif isinstance(self.observation_space, Dict):
            obs_spaces = copy.copy(self.observation_space.spaces)
            for k, v in obs_spaces.items():
                obs_spaces[k] = Box(v.low, v.high, v.shape)
            self.observation_space = Dict(obs_spaces)
        else:
            raise NotImplementedError

    def observation(self, observation: np.ndarray) -> np.ndarray:
        if isinstance(observation, np.ndarray):
            return observation.astype(np.float32)
        elif isinstance(observation, dict):
            observation = copy.copy(observation)
            for k, v in observation.items():
                observation[k] = v.astype(np.float32)
            return observation

class EpisodeMonitor(gym.ActionWrapper):
    """A class that computes episode returns and lengths."""
    def __init__(self, env: gym.Env):
        super().__init__(env)
        self._reset_stats()
        self.total_timesteps = 0

    def _reset_stats(self):
        self.reward_sum = 0.0
        self.episode_length = 0
        self.start_time = time.time()

    def step(self, action: np.ndarray) -> TimeStep:
        observation, reward, terminated, truncated, info = self.env.step(action)

        self.reward_sum += reward
        self.episode_length += 1
        self.total_timesteps += 1
        info['total'] = {'timesteps': self.total_timesteps}

        if terminated or truncated:
            info['episode'] = {}
            info['episode']['return'] = self.reward_sum
            info['episode']['length'] = self.episode_length
            info['episode']['duration'] = time.time() - self.start_time

            if hasattr(self, 'get_normalized_score'):
                info['episode']['return'] = self.get_normalized_score(info['episode']['return']) * 100.0

        return observation, reward, terminated, truncated, info

    #def reset(self) -> np.ndarray:
    #    self._reset_stats()
    #    return self.env.reset()
    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        self._reset_stats()
        return self.env.reset(seed=seed, options=options
            )
class VideoRecorder(gym.Wrapper):
    def __init__(self, env: gym.Env, save_folder: str = '', height: int = 128, width: int = 128, fps: int = 30):
        super().__init__(env)

        self.current_episode = 0
        self.save_folder = save_folder
        self.height = height
        self.width = width
        self.fps = fps
        self.frames = []

        try:
            os.makedirs(save_folder, exist_ok=True)
        except:
            pass

    def render(self, *args, **kwargs):
        """
        Override render to forward height/width parameters to the underlying env.
        """
        if "height" not in kwargs:
            kwargs["height"] = self.height
        if "width" not in kwargs:
            kwargs["width"] = self.width
        return self.env.unwrapped.render(*args, **kwargs)


    def step(self, action: np.ndarray) -> TimeStep:
        #frame = self.env.render(mode='rgb_array', height=self.height, width=self.width)
        frame = self.render(height=self.height, width=self.width)
        if frame is None:
            try:
                frame = self.sim.render(width=self.width,
                                        height=self.height,
                                        mode='offscreen')
                frame = np.flipud(frame)
            except:
                raise NotImplementedError('Rendering is not implemented.')

        self.frames.append(frame)

        observation, reward, done, truncated, info = self.env.step(action)

        if done or truncated:
            save_file = os.path.join(self.save_folder,
                                     f'{self.current_episode}.mp4')
            imageio.mimsave(save_file, self.frames, fps=self.fps)
            self.frames = []
            self.current_episode += 1

        return observation, reward, done, truncated, info


class StickyActionEnv(gym.Wrapper):
    def __init__(self, env, p=0.25):
        super().__init__(env)
        self.p = p
        self.last_action = 0

    def step(self, action):
        if np.random.uniform() < self.p:
            action = self.last_action
        self.last_action = action
        obs, reward, done, truncated, info = self.env.step(action)
        return obs, reward, done, truncated, info


#make env (adapted only for mujoco continuous action spaces)

def make_env(env_name: str, 
            seed: int,
            save_folder: Optional[str] = None,
             add_episode_monitor: bool = False,
             sticky: bool = False,
             flatten: bool = True) -> gym.Env:

    # Check if the env is in gym.
    #all_envs = gym.envs.registry.all()
    all_envs = gym.envs.registry.values()
    env_ids = [env_spec.id for env_spec in all_envs]
    #create DM control env
    domain_name, task_name = env_name.split('-')
    env = DMCEnv(domain_name=domain_name, task_name=task_name, task_kwargs={'random': seed})

    if flatten and isinstance(env.observation_space, gym.spaces.Dict):
        env = gym.wrappers.FlattenObservation(env)

    if add_episode_monitor:
        env = EpisodeMonitor(env)

    env = RescaleAction(env, -1.0, 1.0)

    if save_folder is not None:
        env = VideoRecorder(env, save_folder=save_folder)

    env = SinglePrecision(env)

    if sticky:
        env = StickyActionEnv(env)
    
    #env.seed(seed)
    env.action_space.seed(seed)
    env.observation_space.seed(seed)

    return env



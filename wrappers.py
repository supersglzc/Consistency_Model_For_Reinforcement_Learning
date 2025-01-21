import torch
import numpy as np
import gymnasium


class DMCEnvWrapper:
    def __init__(self, env):
        self.env = env
        self.keys = list(self.env.observation_space.keys())
        dim = 0
        for key in self.keys:
            if key == 'head_height': # account for humanoid head
                dim += 1
            else:
                dim += self.env.observation_space[key].shape[-1]
        self.observation_dim = dim
        self.action_dim = self.env.action_space.shape[-1]
        self.observation_space = gymnasium.spaces.Box(low=-np.inf, high=np.inf, shape=(1, self.observation_dim), dtype=self.env.observation_space[self.keys[0]].dtype)
        self.single_observation_space = gymnasium.spaces.Box(low=-np.inf, high=np.inf, shape=(self.observation_dim,), dtype=self.env.observation_space[self.keys[0]].dtype)
        self.action_space = self.env.action_space
        self.single_action_space = gymnasium.spaces.Box(low=-1, high=1, shape=(self.action_dim,), dtype=self.env.action_space.dtype)
        self.max_episode_length = 1000
        self.device = torch.device("cuda:0")

    def reset(self):
        ob, info = self.env.reset()
        ob = self.cast_obs(ob)
        return ob

    def step(self, actions):
        # actions = actions.cpu().numpy()
        next_obs, rewards, terminated, truncated, infos = self.env.step(actions)
        next_obs = self.cast_obs(next_obs)
        dones = np.logical_or(terminated, truncated)
        timeout = torch.tensor(truncated).bool().to(self.device)
        success = torch.tensor(terminated).to(self.device)
        info_ret = {'time_outs': timeout, 'success': success}

        return next_obs, rewards, dones, info_ret
    
    def cast_obs(self, ob):
        obs = []
        for key in self.keys:
            if key == 'head_height':
                obs.append(np.asarray(ob[key]).reshape(-1, 1))
            else:
                obs.append(ob[key])
        ob = np.concatenate(obs, axis=-1)
        return ob
    

class GYMEnvWrapper:
    def __init__(self, env):
        self.env = env
        self.observation_space = gymnasium.spaces.Box(low=-np.inf, high=np.inf, shape=(self.env.observation_space.shape[-1],), dtype=self.env.observation_space.dtype)
        self.action_space = gymnasium.spaces.Box(low=-1, high=1, shape=(self.env.action_space.shape[-1],), dtype=self.env.action_space.dtype)
        self.max_episode_length = 1000
        self.device = torch.device("cuda:0")

    def reset(self):
        ob = self.env.reset()
        return ob.reshape(1, -1)
    
    def step(self, actions):
        actions = actions.reshape(-1)
        next_obs, rewards, dones, infos = self.env.step(actions)
        return next_obs.reshape(1, -1), rewards.reshape(1, -1), np.array(dones).reshape(1, -1), infos
        
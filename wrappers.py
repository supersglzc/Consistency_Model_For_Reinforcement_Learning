import torch
import numpy as np
import gymnasium


class DMCEnvWrapper:
    def __init__(self, env, num_envs):
        self.env = env
        self.num_envs = num_envs
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
                if self.num_envs > 1:
                    obs.append(np.asarray(ob[key]).reshape(-1, 1))
                else:
                    obs.append(np.asarray(ob[key]).reshape(-1))
            else:
                obs.append(ob[key])
        ob = np.concatenate(obs, axis=-1)
        return ob
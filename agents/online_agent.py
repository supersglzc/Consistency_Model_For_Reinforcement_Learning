import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR
from utils.logger import logger

from agents.diffusion import Diffusion
from agents.consistency import Consistency
from agents.model import MLP
from agents.helpers import EMA


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(Critic, self).__init__()
        self.q1_model = nn.Sequential(nn.Linear(state_dim + action_dim, hidden_dim),
                                      nn.Mish(),
                                      nn.Linear(hidden_dim, hidden_dim),
                                      nn.Mish(),
                                      nn.Linear(hidden_dim, hidden_dim),
                                      nn.Mish(),
                                      nn.Linear(hidden_dim, 1))

        self.q2_model = nn.Sequential(nn.Linear(state_dim + action_dim, hidden_dim),
                                      nn.Mish(),
                                      nn.Linear(hidden_dim, hidden_dim),
                                      nn.Mish(),
                                      nn.Linear(hidden_dim, hidden_dim),
                                      nn.Mish(),
                                      nn.Linear(hidden_dim, 1))

    def forward(self, state, action):
        x = torch.cat([state, action], dim=-1)
        return self.q1_model(x), self.q2_model(x)

    def q1(self, state, action):
        x = torch.cat([state, action], dim=-1)
        return self.q1_model(x)

    def q_min(self, state, action):
        q1, q2 = self.forward(state, action)
        return torch.min(q1, q2)


class DistributionalCritic(nn.Module):
    def __init__(self, state_dim, act_dim, v_min=-10, v_max=10, num_atoms=101, hidden_dim=256, device="cuda"):
        super().__init__()
        self.device = device
        self.net_q1 = nn.Sequential(nn.Linear(state_dim + act_dim, hidden_dim),
                                      nn.Mish(),
                                      nn.Linear(hidden_dim, hidden_dim),
                                      nn.Mish(),
                                      nn.Linear(hidden_dim, hidden_dim),
                                      nn.Mish(),
                                      nn.Linear(hidden_dim, num_atoms))
        self.net_q2 = nn.Sequential(nn.Linear(state_dim + act_dim, hidden_dim),
                                      nn.Mish(),
                                      nn.Linear(hidden_dim, hidden_dim),
                                      nn.Mish(),
                                      nn.Linear(hidden_dim, hidden_dim),
                                      nn.Mish(),
                                      nn.Linear(hidden_dim, num_atoms))

        self.z_atoms = torch.linspace(v_min, v_max, num_atoms, device=device)

    def q_min(self, state, action):
        Q1, Q2 = self.forward(state, action)
        Q1 = torch.sum(Q1 * self.z_atoms.to(self.device), dim=1)
        Q2 = torch.sum(Q2 * self.z_atoms.to(self.device), dim=1)
        return torch.min(Q1, Q2)  # min Q value

    def forward(self, state, action):
        input_x = torch.cat((state, action), dim=1)
        return torch.softmax(self.net_q1(input_x), dim=1), torch.softmax(self.net_q2(input_x), dim=1)  # two Q values

    def q1(self, state, action):
        input_x = torch.cat((state, action), dim=1)
        return torch.softmax(self.net_q1(input_x), dim=1)

def projection(next_dist, reward, done, gamma, v_min=-10, v_max=10, num_atoms=51, support=None, device="cuda:0"):
    delta_z = (v_max - v_min) / (num_atoms - 1)
    batch_size = reward.shape[0]

    target_z = (reward + (1 - done) * gamma * support).clamp(min=v_min, max=v_max)
    b = (target_z - v_min) / delta_z
    l = b.floor().long()
    u = b.ceil().long()

    l[torch.logical_and((u > 0), (l == u))] -= 1
    u[torch.logical_and((l < (num_atoms - 1)), (l == u))] += 1

    proj_dist = torch.zeros_like(next_dist)
    offset = torch.linspace(0, (batch_size - 1) * num_atoms, batch_size, device=device).unsqueeze(1).expand(batch_size, num_atoms).long()
    proj_dist.view(-1).index_add_(0, (l + offset).view(-1), (next_dist * (u.float() - b)).view(-1))
    proj_dist.view(-1).index_add_(0, (u + offset).view(-1), (next_dist * (b - l.float())).view(-1))
    return proj_dist


class OnlineAgent(object):
    def __init__(self,
                 state_dim,
                 action_dim,
                 max_action,
                 model,
                 device,
                 discount,
                 tau,
                 max_q_backup=False,
                 beta_schedule='linear',
                 n_timesteps=100,
                 ema_decay=0.995,
                 step_start_ema=1000,
                 update_ema_every=5,
                 lr=3e-4,
                 lr_decay=False,
                 lr_maxt=1000,
                 grad_norm=1.0,
                 vmin=0,
                 vmax=10,
                 ):

        self.model = MLP(state_dim=state_dim, action_dim=action_dim, device=device)

        if model == 'diffusion':
            self.actor = Diffusion(state_dim=state_dim, action_dim=action_dim, model=self.model, max_action=max_action,
                               beta_schedule=beta_schedule, n_timesteps=n_timesteps,).to(device)
        elif model == 'consistency':
            self.actor = Consistency(state_dim=state_dim, action_dim=action_dim, model=self.model, max_action=max_action,
                               n_timesteps=n_timesteps,).to(device)
        else:
            raise NotImplementedError

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr)

        self.lr_decay = lr_decay
        self.grad_norm = grad_norm

        self.step = 0
        self.step_start_ema = step_start_ema
        self.ema = EMA(ema_decay)
        self.ema_model = copy.deepcopy(self.actor)
        self.update_ema_every = update_ema_every

        self.vmin = vmin
        self.vmax = vmax
        # self.critic = Critic(state_dim, action_dim).to(device)
        self.critic = DistributionalCritic(state_dim, action_dim, v_min=vmin, v_max=vmax).to(device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=3e-4)

        if lr_decay:
            self.actor_lr_scheduler = CosineAnnealingLR(self.actor_optimizer, T_max=lr_maxt, eta_min=0.)
            self.critic_lr_scheduler = CosineAnnealingLR(self.critic_optimizer, T_max=lr_maxt, eta_min=0.)

        self.state_dim = state_dim
        self.max_action = max_action
        self.action_dim = action_dim
        self.discount = discount
        self.tau = tau
        self.device = device
        self.max_q_backup = max_q_backup

    def step_ema(self):
        if self.step < self.step_start_ema:
            return
        self.ema.update_model_average(self.ema_model, self.actor)

    def train(self, replay_buffer, iterations, batch_size=100, log_writer=None):

        metric = {'actor_loss': [], 'critic_loss': []}
        for _ in range(iterations):
            # Sample replay buffer / batch
            data = replay_buffer.sample(batch_size)
            state = data.observations
            action = data.actions
            next_state = data.next_observations
            reward = data.rewards
            not_done = 1 - data.dones

            """ Q Training """
            state = state.float()
            next_state = next_state.float()
            """ Q Training """
            current_q1, current_q2 = self.critic(state, action)

            if self.max_q_backup:
                next_state_rpt = torch.repeat_interleave(next_state, repeats=10, dim=0)
                next_action_rpt = self.ema_model(next_state_rpt)
                target_q1, target_q2 = self.critic_target(next_state_rpt, next_action_rpt)
                target_q1 = target_q1.view(batch_size, 10).max(dim=1, keepdim=True)[0]
                target_q2 = target_q2.view(batch_size, 10).max(dim=1, keepdim=True)[0]
                target_q = torch.min(target_q1, target_q2)
            else:
                next_action = self.ema_model(next_state)
                target_q1, target_q2 = self.critic_target(next_state, next_action)
                # target_q = torch.min(target_q1, target_q2)
                done = (1 - not_done).float()
                target_Q1_projected = projection(next_dist=target_q1,
                                                    reward=reward,
                                                    done=done,
                                                    gamma=self.discount,
                                                    v_min=self.vmin,
                                                    v_max=self.vmax,
                                                    num_atoms=101,
                                                    support=self.critic.z_atoms,
                                                    device=self.device)
                target_Q2_projected = projection(next_dist=target_q2,
                                                    reward=reward,
                                                    done=done,
                                                    gamma=self.discount,
                                                    v_min=self.vmin,
                                                    v_max=self.vmax,
                                                    num_atoms=101,
                                                    support=self.critic.z_atoms,
                                                    device=self.device)
                target_q = torch.min(target_Q1_projected, target_Q2_projected)

            # target_q = (reward + not_done * self.discount * target_q).detach()

            # critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)
            critic_loss = F.binary_cross_entropy(current_q1, target_q) + F.binary_cross_entropy(current_q2, target_q)

            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            if self.grad_norm > 0:
                critic_grad_norms = nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=self.grad_norm, norm_type=2)
            self.critic_optimizer.step()

            """ Policy Training """
            new_action = self.actor(state)

            q1_new_action, q2_new_action = self.critic(state, new_action)
            q1_new_action = torch.sum(q1_new_action * self.critic.z_atoms.to(self.device), dim=1)
            q2_new_action = torch.sum(q2_new_action * self.critic.z_atoms.to(self.device), dim=1)
            if np.random.uniform() > 0.5:
                q_loss = - q1_new_action.mean() / q2_new_action.abs().mean().detach()
            else:
                q_loss = - q2_new_action.mean() / q1_new_action.abs().mean().detach()
            actor_loss = q_loss

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            if self.grad_norm > 0: 
                actor_grad_norms = nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=self.grad_norm, norm_type=2)
            self.actor_optimizer.step()


            """ Step Target network """
            if self.step % self.update_ema_every == 0:
                self.step_ema()

            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            self.step += 1

            """ Log """
            if log_writer is not None:
                if self.grad_norm > 0:
                    log_writer.add_scalar('Actor Grad Norm', actor_grad_norms.max().item(), self.step)
                    log_writer.add_scalar('Critic Grad Norm', critic_grad_norms.max().item(), self.step)
                log_writer.add_scalar('Actor Loss', actor_loss.item(), self.step)
                log_writer.add_scalar('Critic Loss', critic_loss.item(), self.step)
                log_writer.add_scalar('Target_Q Mean', target_q.mean().item(), self.step)

            metric['actor_loss'].append(actor_loss.item())
            metric['critic_loss'].append(critic_loss.item())

        if self.lr_decay: 
            self.actor_lr_scheduler.step()
            self.critic_lr_scheduler.step()

        return metric

    def sample_action(self, state):
        state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
        state_rpt = torch.repeat_interleave(state, repeats=50, dim=0)
        with torch.no_grad():
            action = self.actor.sample(state_rpt)
            q_value = self.critic_target.q_min(state_rpt, action).flatten()
            idx = torch.multinomial(F.softmax(q_value), 1) # sample 1 action with probability proportional to q_value
        return action[idx].cpu().data.numpy().flatten()

    def save_model(self, dir, id=None):
        if id is not None:
            torch.save(self.actor.state_dict(), f'{dir}/actor_{id}.pth')
            torch.save(self.critic.state_dict(), f'{dir}/critic_{id}.pth')
        else:
            torch.save(self.actor.state_dict(), f'{dir}/actor.pth')
            torch.save(self.critic.state_dict(), f'{dir}/critic.pth')

    def load_model(self, dir, id=None):
        if id is not None:
            self.actor.load_state_dict(torch.load(f'{dir}/actor_{id}.pth'))
            self.critic.load_state_dict(torch.load(f'{dir}/critic_{id}.pth'))
        else:
            self.actor.load_state_dict(torch.load(f'{dir}/actor.pth'))
            self.critic.load_state_dict(torch.load(f'{dir}/critic.pth'))



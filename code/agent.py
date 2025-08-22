import torch
import torch.nn as nn
import torch.optim as optim
from replay_buffer import ReplayBuffer
from env_utils import Env
import numpy as np

np.random.seed(42)
torch.manual_seed(42)

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, action_high, action_low, hidden_dim=64):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        
        self.fc_mean = nn.Linear(hidden_dim, action_dim)
        self.fc_std = nn.Linear(hidden_dim, action_dim)
        
        self.activation = nn.ReLU()
        self.register_buffer('action_high', torch.from_numpy(action_high).float())
        self.register_buffer('action_low', torch.from_numpy(action_low).float())
        self.action_dim = action_dim
        self.optimizer = optim.Adam(self.parameters(), lr=1e-4)

    def forward(self, state):
        x = self.activation(self.fc1(state))
        x = self.activation(self.fc2(x))
        
        mean = self.fc_mean(x)
        std = self.fc_std(x)
        std = torch.clamp(std, min=-20, max=2)
        std = torch.exp(std)
        
        dist = torch.distributions.Normal(mean, std)
        
        action_raw = dist.rsample()
        action_squashed = torch.tanh(action_raw)
        
        log_prob = dist.log_prob(action_raw)
        log_prob -= torch.log(1 - action_squashed.pow(2) + 1e-6)
        log_prob = log_prob.sum(axis=-1)
        
        action = self.action_low + (action_squashed + 1.0) * 0.5 * (self.action_high - self.action_low)
        return action, log_prob


class Critic(nn.Module):
    def __init__(self, state_dim, hidden_dim=64):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)
        self.activation = nn.ReLU()
        self.optimizer = optim.Adam(self.parameters(), lr=1e-4)
        
    def forward(self, state):
        x = self.activation(self.fc1(state))
        x = self.activation(self.fc2(x))
        x = self.fc3(x)
        return x

class Agent:
    def __init__(self, memory: ReplayBuffer, env: Env, actor: Actor, critic: Critic):
        self.memory = memory
        self.lr = 1e-4
        self.gamma = 0.99
        self.batch_size = 1024
        self.action_dim = env.get_action_space().shape[0]
        self.state_dim = env.get_observation_space().shape[0]
        self.action_high = env.get_action_space().high
        self.action_low = env.get_action_space().low
        self.actor = actor
        self.critic = critic
        self.env = env
       
        
    def get_action(self, state):
        return self.actor(state)

    def get_value(self, state):
        return self.critic(state)
    
    def get_action_and_value(self, state):
        action, log_prob = self.actor(state)
        value = self.critic(state)
        return action, log_prob, value


if __name__ == "__main__":
    env = Env('Walker2d-v5', 10, 42)
    memory = ReplayBuffer(100000)
    state_dim = env.get_observation_space().shape[0]
    action_dim = env.get_action_space().shape[0]
    action_high = env.get_action_space().high
    action_low = env.get_action_space().low
    
    actor = Actor(state_dim, action_dim, action_high, action_low)
    critic = Critic(state_dim)
    agent = Agent(memory, env, actor, critic)
    obs, _ = env.reset(seed=42)
    action, log_prob, value = agent.get_action_and_value(torch.from_numpy(obs).float())
    print(action)
    print(log_prob)
    print(value)
        
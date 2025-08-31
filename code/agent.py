import torch
import torch.nn as nn
from env_utils import Env
import numpy as np
from state_encoder import NoEncoder

np.random.seed(42)
torch.manual_seed(42)


class Actor(nn.Module):
    def __init__(self, action_dim, action_high, action_low, hidden_dim=64, state_encoder=nn.Identity(), discrete_action=False):
        super(Actor, self).__init__()
        self.state_encoder = state_encoder
        self.fc1 = nn.Linear(state_encoder.get_state_dim(), hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        
        if discrete_action:
            self.fc_logits = nn.Linear(hidden_dim, action_dim)
        else:
            self.fc_mean = nn.Linear(hidden_dim, action_dim)
            self.fc_std = nn.Linear(hidden_dim, action_dim)
            torch.nn.init.constant_(self.fc_std.weight, 0)
            torch.nn.init.constant_(self.fc_std.bias, 0)
            
        self.activation = nn.ReLU()
        if action_high is not None:
            self.register_buffer('action_high', torch.from_numpy(action_high).float())
        if action_low is not None:
            self.register_buffer('action_low', torch.from_numpy(action_low).float())
        self.action_dim = action_dim
        self.discrete_action = discrete_action
        
    def forward_continuous(self, state):
        state = self.state_encoder(state)
        x = self.activation(self.fc1(state))
        x = self.activation(self.fc2(x))
        
        mean = self.fc_mean(x)
        std = self.fc_std(x)
        std = torch.clamp(std, min=-40, max=4)
        std = torch.exp(std)
        
        dist = torch.distributions.Normal(mean, std)
        return dist

    def forward_discrete(self, state):
        state = self.state_encoder(state)
        x = self.activation(self.fc1(state))
        x = self.activation(self.fc2(x))
        x = self.fc_logits(x)
        dist = torch.distributions.Categorical(logits=x)
        return dist

    def forward(self, state):
        if self.discrete_action:
            return self.forward_discrete(state)
        else:
            return self.forward_continuous(state)

    def get_action(self, state, action=None):
        dist = self.forward(state)
                
        if action is None:
            action = dist.sample()
        log_prob = dist.log_prob(action)
        log_prob = log_prob.sum(axis=-1)
        entropy = dist.entropy().sum(axis=-1)
        return action, log_prob, entropy, dist


class Critic(nn.Module):
    def __init__(self, hidden_dim=64, state_encoder=nn.Identity()):
        super(Critic, self).__init__()
        self.state_encoder = state_encoder
        self.fc1 = nn.Linear(state_encoder.get_state_dim(), hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)
        self.activation = nn.ReLU()
        
    def forward(self, state):
        state = self.state_encoder(state)
        x = self.activation(self.fc1(state))
        x = self.activation(self.fc2(x))
        x = self.fc3(x)
        return x

class Agent:
    def __init__(self, actor: Actor, critic: Critic):
        self.actor = actor
        self.critic = critic
        
    def get_action(self, state):
        return self.actor(state)

    def get_value(self, state):
        return self.critic(state)
    
    def get_action_and_value(self, state, action=None):
        action, log_prob, entropy, dist = self.actor.get_action(state, action)
        value = self.critic(state).squeeze()
        return action, log_prob, entropy, value, dist


if __name__ == "__main__":
    env = Env('Walker2d-v5', 10, 42)
    state_dim = env.get_observation_space().shape[0]
    action_dim = env.get_action_space().shape[0]
    action_high = env.get_action_space().high
    action_low = env.get_action_space().low
    
    actor = Actor(action_dim, action_high, action_low, state_encoder=NoEncoder(state_dim))
    critic = Critic(state_encoder=NoEncoder(state_dim))
    agent = Agent(actor, critic)
    obs, _ = env.reset(seed=42)
    action, log_prob, entropy, value = agent.get_action_and_value(torch.from_numpy(obs).float())
    print(action)
    print(log_prob)
    print(entropy)
    print(value)
        
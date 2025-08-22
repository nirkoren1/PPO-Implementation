import torch
import torch.nn as nn
import torch.optim as optim
from agent import Actor, Critic, Agent
from replay_buffer import ReplayBuffer
from env_utils import Env

class PPO:
    def __init__(self, env: Env, num_envs: int, num_steps: int, num_episodes: int):
        self.env = env
        self.agent = Agent(ReplayBuffer(100000), 
                             env, 
                             Actor(env.get_observation_space().shape[0],
                                   env.get_action_space().shape[0],
                                   env.get_action_space().high,
                                   env.get_action_space().low), 
                             Critic(env.get_observation_space().shape[0]))
        self.envs = Env(env.env_name, num_envs, env.seed)
        self.num_steps = num_steps
        self.num_episodes = num_episodes
        
    def learn(self):
        pass
        
    def train(self):
        pass





if __name__ == "__main__":
    env = Env('Walker2d-v5', 10, 42)
    ppo = PPO(env, 10, 10, 10, 10)
    ppo.train()
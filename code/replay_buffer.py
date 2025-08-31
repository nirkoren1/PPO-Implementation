import numpy as np
import torch

np.random.seed(42)

class ReplayBuffer:
    def __init__(self, num_steps, num_envs, obs_shape, action_shape, gamma, gae_lambda, discrete_action):
        self.states = torch.zeros((num_steps, num_envs) + obs_shape)
        self.actions = torch.zeros((num_steps, num_envs) + action_shape, dtype=torch.long if discrete_action else torch.float)
        self.log_probs = torch.zeros((num_steps, num_envs))
        self.rewards = torch.zeros((num_steps, num_envs))
        self.dones = torch.zeros((num_steps, num_envs))
        self.terminations = torch.zeros((num_steps, num_envs))
        self.values = torch.zeros((num_steps, num_envs))
        self.action_means = torch.zeros((num_steps, num_envs) + action_shape)
        self.action_stds = torch.zeros((num_steps, num_envs) + action_shape)
        self.action_shape = action_shape
        self.obs_shape = obs_shape
        self.num_steps = num_steps
        self.num_envs = num_envs
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.position = 0
        self.size = 0
        
        
    def push(self, step_records, step):
        for key, value in step_records.items():
            if key == 'states':
                self.states[step] = value
            elif key == 'actions':
                self.actions[step] = value
            elif key == 'log_probs':
                self.log_probs[step] = value
            elif key == 'rewards':
                self.rewards[step] = value
            elif key == 'dones':
                self.dones[step] = value
            elif key == 'terminations':
                self.terminations[step] = value
            elif key == 'values':
                self.values[step] = value
            elif key == 'action_means':
                self.action_means[step] = value
            elif key == 'action_stds':
                self.action_stds[step] = value
            else:
                raise ValueError(f"Invalid key: {key}")
            
    def calculate_advantages(self, next_value, next_termination):
        advantages = torch.zeros_like(self.rewards)
        last_gae_lam = 0
        for t in reversed(range(self.num_steps)):
            if t == self.num_steps - 1:
                next_non_terminal = 1.0 - next_termination
                next_values = next_value
            else:
                next_non_terminal = 1.0 - self.terminations[t+1]
                next_values = self.values[t+1]
            
            delta = self.rewards[t] + self.gamma * next_values * next_non_terminal - self.values[t]
            advantages[t] = last_gae_lam = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae_lam
        returns = advantages + self.values
        self.advantages = advantages
        self.returns = returns

    def get_batches(self, batch_size):
        b_states = self.states.reshape((-1,) + self.obs_shape)
        b_actions = self.actions.reshape((-1,) + self.action_shape)
        b_log_probs = self.log_probs.reshape(-1)
        b_advantages = self.advantages.reshape(-1)
        b_returns = self.returns.reshape(-1)
        b_means = self.action_means.reshape((-1,) + self.action_shape)
        b_stds = self.action_stds.reshape((-1,) + self.action_shape)
        
        indices = np.random.permutation(self.num_steps * self.num_envs)
        
        for i in range(0, self.num_steps * self.num_envs, batch_size):
            minibatch_indices = indices[i:i+batch_size]
            yield (b_states[minibatch_indices],
                   b_actions[minibatch_indices],
                   b_log_probs[minibatch_indices],
                   b_advantages[minibatch_indices],
                   b_returns[minibatch_indices],
                   b_means[minibatch_indices],
                   b_stds[minibatch_indices])
    
    
if __name__ == "__main__":
    memory = ReplayBuffer(1000)
    for i in range(1000):
        memory.push(i, i, i, i, i, i, i)
    print(memory.values)
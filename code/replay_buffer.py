import numpy as np

np.random.seed(42)

class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.rewards = np.array([None] * capacity)
        self.actions = np.array([None] * capacity)
        self.states = np.array([None] * capacity)
        self.next_states = np.array([None] * capacity)
        self.terminateds = np.array([None] * capacity)
        self.probs = np.array([None] * capacity)
        self.values = np.array([None] * capacity)
        self.position = 0
        self.size = 0
        
    def push(self, reward, action, state, next_state, terminated, prob, value):
        self.rewards[self.position] = reward
        self.actions[self.position] = action
        self.states[self.position] = state
        self.next_states[self.position] = next_state
        self.terminateds[self.position] = terminated
        self.probs[self.position] = prob
        self.values[self.position] = value
        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def get_batches(self, batch_size):
        batches = []
        indices = np.arange(0, self.capacity)
        indices = np.random.permutation(indices)
        for i in range(0, self.capacity, batch_size):
            batch_indices = indices[i:i+batch_size]
            batches.append((self.rewards[batch_indices],
                            self.actions[batch_indices],
                            self.states[batch_indices],
                            self.next_states[batch_indices],
                            self.terminateds[batch_indices],
                            self.probs[batch_indices]))
        return iter(batches)

    def __len__(self):
        return self.size
    
    def clear(self):
        self.position = 0
        self.size = 0
    
    
if __name__ == "__main__":
    memory = ReplayBuffer(1000)
    for i in range(1000):
        memory.push(i, i, i, i, i, i, i)
    print(memory.values)
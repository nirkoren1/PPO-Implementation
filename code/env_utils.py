import gymnasium as gym

class Env:
    def __init__(self, env_name, num_envs, seed):
        self.env_name = env_name
        self.num_envs = num_envs
        self.seed = seed
        self.envs = gym.vector.SyncVectorEnv([self.make_env(i) for i in range(num_envs)])
        
    def make_env(self, rank):
        def _init():
            env = gym.make(self.env_name)
            return env
        return _init
    
    def step(self, action):
        return self.envs.step(action)
    
    def reset(self, seed=None):
        return self.envs.reset(seed=seed)
    
    def close(self):
        self.envs.close()
        
    def get_action_space(self):
        return self.envs.single_action_space
    
    def get_observation_space(self):
        return self.envs.single_observation_space
    
    def get_num_envs(self):
        return self.num_envs
    
if __name__ == "__main__":
    env = Env('Walker2d-v5', 10, 42)
    # get the max and min of the action space
    print(env.get_action_space().high)
    print(env.get_action_space().low)
    # get the max and min of the observation space
    print(env.get_observation_space().high)
    print(env.get_observation_space().low)
    print(env.get_num_envs())
    env.close()
import gymnasium as gym
import ale_py
from ale_py.vector_env import AtariVectorEnv
from stable_baselines3.common.atari_wrappers import (
    ClipRewardEnv,
    EpisodicLifeEnv,
    FireResetEnv,
    MaxAndSkipEnv,
    NoopResetEnv,
)

def make_env(gym_id, seed):
    def thunk():
        env = gym.make(gym_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env = NoopResetEnv(env, noop_max=30)
        env = MaxAndSkipEnv(env, skip=4)
        env = EpisodicLifeEnv(env)
        if "FIRE" in env.unwrapped.get_action_meanings():
            env = FireResetEnv(env)
        env = ClipRewardEnv(env)
        env = gym.wrappers.ResizeObservation(env, (84, 84))
        env = gym.wrappers.GrayscaleObservation(env)
        env = gym.wrappers.FrameStackObservation(env, 4)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        return env

    return thunk

class Env:
    def __init__(self, env_name, num_envs, seed, atari=False):
        self.env_name = env_name
        self.num_envs = num_envs
        self.seed = seed
        self.atari = atari
        if atari:
            # self.envs = AtariVectorEnv(env_name, num_envs=num_envs, thread_affinity_offset=0)
            self.envs = gym.vector.SyncVectorEnv(
                [make_env(env_name, seed + i) for i in range(num_envs)]
            )
            # self.envs = gym.wrappers.vector.RecordEpisodeStatistics(self.envs)
        else:
            self.envs = gym.make_vec(env_name, num_envs=num_envs)
            self.envs = gym.wrappers.vector.RecordEpisodeStatistics(self.envs)

    def step(self, action):
        return self.envs.step(action)
    
    def reset(self, seed=None):
        if seed is None:
            seed = self.seed
        return self.envs.reset(seed=seed)
    
    def close(self):
        self.envs.close()
        
    def get_action_space(self):
        return self.envs.single_action_space
    
    def get_observation_space(self):
        return self.envs.single_observation_space
    
    def get_num_envs(self):
        return self.num_envs
    
    def get_env_name(self):
        return self.env_name

if __name__ == "__main__":
    atari_env = Env('ALE/Breakout-v5', 4, 42)
    print(f"Environment Name: {atari_env.get_env_name()}")
    print(f"Action Space: {atari_env.get_action_space()}")
    print(f"Observation Space Shape: {atari_env.get_observation_space().shape}")
    atari_env.close()

    classic_env = Env('Walker2d-v5', 10, 42)
    print(f"\nEnvironment Name: {classic_env.get_env_name()}")
    print(f"Action Space High: {classic_env.get_action_space().high}")
    print(f"Action Space Low: {classic_env.get_action_space().low}")
    print(f"Observation Space Shape: {classic_env.get_observation_space().shape}")
    classic_env.close()
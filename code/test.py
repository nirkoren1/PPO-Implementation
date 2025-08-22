import gymnasium as gym
import time

# Create a vectorized environment to run multiple instances of Walker2d-v5 in parallel.
# This significantly speeds up data collection compared to a single environment.
num_envs = 1  # Number of parallel environments
env = gym.make_vec('Walker2d-v5', num_envs=num_envs, vectorization_mode='sync')

# Reset the environments to get the initial observations
observation, info = env.reset()
print(env.action_space)
print(env.observation_space)

# start_time = time.time()

# # Run the environments for a fixed number of total steps
# for _ in range(100000 // num_envs):
#     # Sample a batch of random actions, one for each environment
#     action = env.action_space.sample()
    
#     # Apply the actions to the environments
#     # This returns batches of observations, rewards, etc.
#     observation, reward, terminated, truncated, info = env.step(action)
    
#     # Vectorized environments automatically reset upon termination or truncation,
#     # so there is no need to manually check and reset the environment.

# end_time = time.time()
# print(f"Vectorized environment execution time: {end_time - start_time:.2f} seconds")

# Close the environments
env.close()

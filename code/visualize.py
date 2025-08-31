import gymnasium as gym
import torch
from agent import Actor
import time
import argparse
from state_encoder import NoEncoder

def parse_args():
    parser = argparse.ArgumentParser(description="Visualize a trained PPO agent for classic envs.")
    parser.add_argument("--env-name", type=str, default="Walker2d-v5", help="The name of the mujoco environment, 'Walker2d-v5', 'HalfCheetah-v5', 'Hopper-v5', 'InvertedDoublePendulum-v5', 'InvertedPendulum-v5', 'Reacher-v5', 'Swimmer-v5'.")
    parser.add_argument("--model-path", type=str, default="code/models/best_actor_Clip_Loss_Walker2d-v5.pth", help="Path to the saved actor model weights")
    parser.add_argument("--seed", type=int, default=42, help="Seed for the environment")
    parser.add_argument("--num-episodes", type=int, default=10, help="Number of episodes to visualize")
    return parser.parse_args()

def visualize_agent(env_name, model_path, episodes=1):
    env = gym.make(env_name, render_mode="human")
    
    action_dim = env.action_space.shape[0]
    action_high = env.action_space.high
    action_low = env.action_space.low
    state_dim = env.observation_space.shape

    actor = Actor(action_dim, action_high, action_low, state_encoder=NoEncoder(state_dim))
    
    try:
        actor.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    except FileNotFoundError:
        print(f"Model file not found at {model_path}")
        env.close()
        exit()
        
    actor.eval()

    for i in range(episodes):
        obs, info = env.reset()
        done = False
        truncated = False
        total_reward = 0
        
        while not done and not truncated:
            
            obs_tensor = torch.from_numpy(obs).float().unsqueeze(0)
            
            with torch.no_grad():
                action, _, _, _ = actor.get_action(obs_tensor)
            
            obs, reward, done, truncated, info = env.step(action.squeeze(0).cpu().numpy())
            
            total_reward += reward
            
            time.sleep(0.01)

        print(f"Episode {i+1}: Total Reward = {total_reward:.2f}")

    env.close()

if __name__ == "__main__":
    args = parse_args()

    ENV_NAME = args.env_name
    MODEL_PATH = args.model_path
    
    visualize_agent(ENV_NAME, MODEL_PATH, args.num_episodes)
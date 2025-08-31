import torch
import gymnasium as gym
from gymnasium import wrappers
import numpy as np
import argparse
import time
from agent import Actor
from state_encoder import AtariImageEncoder, AtariNoEncoder


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize a trained PPO agent for Atari envs.")
    parser.add_argument("--env-name", type=str, default="ALE/Alien-v5", help="The name of the Atari environment, 'ALE/Alien-v5', 'ALE/Breakout-v5'.")
    parser.add_argument("--model-path", type=str, default="code/models/best_actor_Clip_Loss_ALE_Alien-v5.pth", help="Path to the saved actor model weights")
    parser.add_argument("--seed", type=int, default=42, help="Seed for the environment")
    parser.add_argument("--num-episodes", type=int, default=1, help="Number of episodes to visualize")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()

    try:
        env = gym.make(args.env_name, render_mode="human")
    except Exception as e:
        print(e)
        exit()

    env = wrappers.AtariPreprocessing(env, frame_skip=1)
    env = wrappers.FrameStackObservation(env, 4)


    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    obs_space_shape = env.observation_space.shape
    action_space_dim = env.action_space.n

    state_encoder = AtariImageEncoder(obs_space_shape)
    actor = Actor(
        action_dim=action_space_dim,
        action_high=None,
        action_low=None,
        state_encoder=state_encoder,
        discrete_action=True
    )

    try:
        actor.load_state_dict(torch.load(args.model_path, map_location=torch.device('cpu')))
    except FileNotFoundError:
        print(f"Model file not found at {args.model_path}")
        env.close()
        exit()
    except Exception as e:
        print(f"Error while loading the model: {e}")
        env.close()
        exit()

    actor.eval()
    
    actions = set()

    for episode in range(args.num_episodes):
        obs, info = env.reset(seed=args.seed + episode)
        obs = np.array(obs)

        terminated = False
        truncated = False
        total_reward = 0
        print(f"--- Starting Episode {episode + 1}/{args.num_episodes} ---")

        while not terminated and not truncated:
            env.render()

            obs_tensor = torch.from_numpy(obs).float().unsqueeze(0)

            with torch.no_grad():
                dist = actor(obs_tensor)
                action = torch.argmax(dist.logits, dim=-1).item()
                actions.add(action)
                print(actions)

            obs, reward, terminated, truncated, info = env.step(action)
            obs = np.array(obs)
            total_reward += reward

            time.sleep(0.02)

        print(f"Episode finished. Total Reward: {total_reward}")

    env.close()
    print("\nVisualization complete.")

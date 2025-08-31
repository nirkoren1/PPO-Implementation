import torch
import torch.optim as optim
from agent import Actor, Critic, Agent
from env_utils import Env
import numpy as np
import time
from collections import deque
import os
import argparse
from losses import Clip_Loss, No_clip_Loss, KL_Penalty_Loss, Value_Loss
from replay_buffer import ReplayBuffer
import pickle
import matplotlib.pyplot as plt
import ale_py
from state_encoder import NoEncoder, AtariImageEncoder, AtariNoEncoder

class PPO:
    def __init__(self, env_name, 
                 num_envs, seed, 
                 loss_fn_class, loss_kwargs, 
                 total_timesteps=200000, num_steps=2048, 
                 verbose=True, discrete_action=False, 
                 state_encoder=NoEncoder, 
                 update_epochs=10, minibatch_size=64,
                 adam_step_size=3e-4, entropy_coef=0.0, value_coef=0.5, max_grad_norm=0.5,
                 gamma=0.99, gae_lambda=0.95, anneal_lr=False, anneal_clip=False,
                 atari=False
                 ):
        self.num_envs = num_envs
        self.num_steps = num_steps
        self.total_timesteps = total_timesteps
        self.batch_size = self.num_envs * self.num_steps
        self.verbose = verbose
        
        self.update_epochs = update_epochs
        self.minibatch_size = minibatch_size
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.ent_coef = entropy_coef
        self.vf_coef = value_coef
        self.lr = adam_step_size
        self.max_grad_norm = max_grad_norm
        self.anneal_lr = anneal_lr
        self.anneal_clip = anneal_clip

        self.envs = Env(env_name, num_envs, seed, atari=atari)
        actor_encoder = state_encoder(self.envs.get_observation_space().shape)
        critic_encoder = state_encoder(self.envs.get_observation_space().shape)
        
        if discrete_action:
            action_dim = self.envs.get_action_space().n
            action_high = None
            action_low = None
        else:
            action_dim = self.envs.get_action_space().shape[0]
            action_high = self.envs.get_action_space().high
            action_low = self.envs.get_action_space().low

        self.actor = Actor(action_dim, action_high, action_low, state_encoder=actor_encoder, discrete_action=discrete_action)
        self.critic = Critic(state_encoder=critic_encoder)
        self.agent = Agent(self.actor, self.critic)
        
        self.loss_fn = loss_fn_class(**loss_kwargs)
        self.v_loss_fn = Value_Loss(vf_coef=self.vf_coef)

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.lr, eps=1e-5)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.lr, eps=1e-5)
        
        # For Adaptive KL Penalty
        self.adaptive_kl = loss_kwargs.get('adaptive_kl', False)
        if self.adaptive_kl:
            self.beta = loss_kwargs['beta']
            self.d_targ = loss_kwargs['d_targ']


    def learn(self):
        for epoch in range(self.update_epochs):            
            for b_states, b_actions, b_log_probs, b_advantages, b_returns, b_means, b_stds in self.replay_buffer.get_batches(self.minibatch_size):
                _, new_log_probs, entropy, new_values, new_dist = self.agent.get_action_and_value(b_states, b_actions)

                log_ratio = new_log_probs - b_log_probs
                ratio = torch.exp(log_ratio)

                kl_div = None
                if isinstance(self.loss_fn, KL_Penalty_Loss):
                    old_dist = torch.distributions.Normal(b_means, b_stds)
                    kl_div = torch.distributions.kl.kl_divergence(old_dist, new_dist).sum(dim=-1)
                
                loss_kwargs = {
                    'ratio': ratio,
                    'advantages': b_advantages,
                    'kl': kl_div
                }
                
                # Actor update
                pg_loss = self.loss_fn(**loss_kwargs)
                entropy_loss = entropy.mean()
                actor_loss = pg_loss - self.ent_coef * entropy_loss

                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
                self.actor_optimizer.step()

                # Critic update
                v_loss = self.v_loss_fn(new_values=new_values, returns=b_returns)

                self.critic_optimizer.zero_grad()
                v_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
                self.critic_optimizer.step()

        # For adaptive KL penalty, update beta
        if self.adaptive_kl:
            self.update_beta()


    def train(self, target_score=None):
        os.makedirs("models", exist_ok=True)
        os.makedirs("reward_plots", exist_ok=True)
        self.replay_buffer = ReplayBuffer(self.num_steps, self.num_envs, self.envs.get_observation_space().shape, self.envs.get_action_space().shape, self.gamma, self.gae_lambda, self.actor.discrete_action)
        
        ep_info_buffer = deque(maxlen=100)
        best_reward = -np.inf
        one_million_reward = -np.inf
        reward_plot = []

        num_updates = self.total_timesteps // self.batch_size
        obs, _ = self.envs.reset()
        next_obs = torch.from_numpy(obs).float()
        next_done = torch.zeros(self.num_envs)
        
        global_step = 0
        start_time = time.time()
        
        for update in range(1, num_updates + 1):
            # Annealing
            if self.anneal_lr or self.anneal_clip:
                frac = 1.0 - (update - 1.0) / num_updates
                if self.anneal_lr:
                    lrnow = frac * self.lr
                    self.actor_optimizer.param_groups[0]["lr"] = lrnow
                    self.critic_optimizer.param_groups[0]["lr"] = lrnow
                if self.anneal_clip and isinstance(self.loss_fn, Clip_Loss):
                    self.loss_fn.clip_coef = frac * self.loss_fn.clip_coef
            
            step_records = {}
            for step in range(self.num_steps):
                global_step += self.num_envs
                
                step_records['states'] = next_obs
                step_records['dones'] = next_done

                with torch.no_grad():
                    action, log_prob, _, value, dist = self.agent.get_action_and_value(next_obs)
                    
                    step_records['values'] = value.squeeze()
                
                step_records['actions'] = action
                step_records['log_probs'] = log_prob
                
                if not self.actor.discrete_action:
                    step_records['action_means'] = dist.mean
                    step_records['action_stds'] = dist.stddev

                
                obs, reward, terminated, truncated, info = self.envs.step(action.cpu().numpy())
                done = np.logical_or(terminated, truncated)
                
                step_records['rewards'] = torch.from_numpy(reward).float()
                step_records['terminations'] = torch.from_numpy(terminated).float()
                next_obs = torch.from_numpy(obs).float()
                next_done = torch.from_numpy(done).float()
                next_termination = torch.from_numpy(terminated).float()

                if "episode" in info.keys():
                    ep_return = info["episode"]["r"]
                    for i in range(len(ep_return)):
                        if info["_episode"][i]:
                            ep_info_buffer.append(ep_return[i].item())
                            
                self.replay_buffer.push(step_records, step)

            # advantages Calculation
            with torch.no_grad():
                next_value = self.agent.critic(next_obs).reshape(1, -1)
            self.replay_buffer.calculate_advantages(next_value, next_termination)
            
            self.learn()

            if len(ep_info_buffer) > 0:
                avg_reward = np.mean(ep_info_buffer)
                reward_plot.append((global_step, avg_reward))
                if avg_reward > one_million_reward and global_step > 1_000_000 and one_million_reward == -np.inf:
                    one_million_reward = avg_reward
                if self.verbose:
                    print(f"Global Step: {global_step}, steps/s: {int(global_step / (time.time() - start_time))}, Avg Reward: {avg_reward:.2f}")
                if target_score is not None and avg_reward > target_score:
                    print(f"Target score reached: {avg_reward}")
                    return avg_reward, reward_plot, global_step, one_million_reward
                elif avg_reward > best_reward:
                    best_reward = avg_reward
                    torch.save(self.actor.state_dict(), f"models/best_actor_{self.loss_fn.__class__.__name__.replace('/', '_')}_{self.envs.get_env_name().replace('/', '_')}.pth")
                    torch.save(self.critic.state_dict(), f"models/best_critic_{self.loss_fn.__class__.__name__.replace('/', '_')}_{self.envs.get_env_name().replace('/', '_')}.pth")
            elif self.verbose:
                print(f"Global Step: {global_step}, steps/s: {int(global_step / (time.time() - start_time))}")

        self.envs.close()
        
        if target_score is not None:
            return best_reward, reward_plot, global_step, one_million_reward
        
        if len(ep_info_buffer) > 0:
            return np.mean(ep_info_buffer), reward_plot
        else:
            return -np.inf, reward_plot

    def update_beta(self):
        with torch.no_grad():
            full_b_states = self.replay_buffer.states.reshape((-1,) + self.envs.get_observation_space().shape)
            full_b_means = self.replay_buffer.action_means.reshape((-1,) + self.envs.get_action_space().shape)
            full_b_stds = self.replay_buffer.action_stds.reshape((-1,) + self.envs.get_action_space().shape)
    
            new_dist = self.agent.actor(full_b_states)
            old_dist = torch.distributions.Normal(full_b_means, full_b_stds)
            mean_kl = torch.distributions.kl.kl_divergence(old_dist, new_dist).sum(axis=-1).mean().item()
                    
            if mean_kl < self.d_targ / 1.5:
                self.beta /= 2
            elif mean_kl > self.d_targ * 1.5:
                self.beta *= 2
                    
            self.loss_fn.beta = self.beta


def parse_args():
    parser = argparse.ArgumentParser()
    # Core parameters
    parser.add_argument("--env-name", type=str, default='Walker2d-v5', help="environment name")
    parser.add_argument("--seed", type=int, default=42, help="seed")
    parser.add_argument("--total-timesteps", type=int, default=10_000_000, help="total timesteps")
    parser.add_argument("--num-envs", type=int, default=1, help="number of parallel environments")

    # PPO algorithm parameters
    parser.add_argument("--num-steps", type=int, default=2048, help="the number of steps to run in each environment per policy rollout aka Horizon T")
    parser.add_argument("--update-epochs", type=int, default=10, help="the K epochs to update the policy")
    parser.add_argument("--minibatch-size", type=int, default=64, help="the mini batch size")
    parser.add_argument("--gamma", type=float, default=0.99, help="the discount factor gamma")
    parser.add_argument("--gae-lambda", type=float, default=0.95, help="the lambda for the general advantage estimation")
    parser.add_argument("--adam-stepsize", type=float, default=3e-4, help="learning rate")
    parser.add_argument("--ent-coef", type=float, default=0.01, help="coefficient of the entropy in the loss function")
    parser.add_argument("--vf-coef", type=float, default=1.0, help="coefficient of the value function in the loss function")
    parser.add_argument("--max-grad-norm", type=float, default=0.5, help="the maximum norm for the gradient clipping")

    # Annealing
    parser.add_argument("--anneal-lr", action="store_true", default=False, help="Enable learning rate annealing")
    parser.add_argument("--anneal-clip", action="store_true", default=False, help="Enable clip coefficient annealing")

    # Agent and Loss parameters
    parser.add_argument("--state-encoder", type=str, default="NoEncoder", choices=["NoEncoder", "AtariImageEncoder", "AtariNoEncoder"], help="state encoder type")
    parser.add_argument("--discrete-action", default=False, action="store_true", help="whether to use discrete action")
    parser.add_argument("--loss-type", type=str, default="clip", choices=["clip", "noclip", "kl_penalty"], help="loss function type")
    parser.add_argument("--clip-coef", type=float, default=0.2, help="surrogate clipping coefficient in the loss function")
    parser.add_argument("--beta", type=float, default=1.0, help="coefficient for the KL penalty in the loss function")
    parser.add_argument("--adaptive-kl", default=False, action="store_true", help="whether to use adaptive KL penalty")
    parser.add_argument("--d-targ", type=float, default=0.01, help="the target KL divergence for adaptive KL penalty")
    parser.add_argument("--atari", default=False, action="store_true", help="whether to use Atari environment")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    
    loss_map = {
        "clip": Clip_Loss,
        "noclip": No_clip_Loss,
        "kl_penalty": KL_Penalty_Loss
    }
    
    state_encoder_map = {
        "NoEncoder": NoEncoder,
        "AtariImageEncoder": AtariImageEncoder,
        "AtariNoEncoder": AtariNoEncoder
    }
    
    loss_kwargs = {
        "clip": {"clip_coef": args.clip_coef},
        "noclip": {},
        "kl_penalty": {"beta": args.beta, "adaptive_kl": args.adaptive_kl, "d_targ": args.d_targ}
    }

    selected_loss_class = loss_map[args.loss_type]
    selected_loss_kwargs = loss_kwargs[args.loss_type]

    ppo = PPO(
        env_name=args.env_name, 
        num_envs=args.num_envs, 
        seed=args.seed,
        total_timesteps=args.total_timesteps,
        loss_fn_class=selected_loss_class,
        loss_kwargs=selected_loss_kwargs,
        state_encoder=state_encoder_map[args.state_encoder],
        discrete_action=args.discrete_action,
        num_steps=args.num_steps,
        update_epochs=args.update_epochs,
        minibatch_size=args.minibatch_size,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        adam_step_size=args.adam_stepsize,
        entropy_coef=args.ent_coef,
        value_coef=args.vf_coef,
        max_grad_norm=args.max_grad_norm,
        anneal_lr=args.anneal_lr,
        anneal_clip=args.anneal_clip,
        atari=args.atari
    )
    final_reward, reward_plot = ppo.train()
    print(f"Training complete. Final average reward: {final_reward}")
    
    # plot the reward plot
    if reward_plot:
        steps, rewards = zip(*reward_plot)
        plt.figure(figsize=(12, 6))
        plt.plot(steps, rewards)
        plt.xlabel("Timesteps")
        plt.ylabel("Average Reward")
        plt.title(f"{args.env_name}")
        plt.grid(True)
        
        plot_filename = f"reward_plots/reward_plot_{args.env_name.replace('/', '_')}.png"
        plt.savefig(plot_filename)
        print(f"Reward plot saved to {plot_filename}")
        
        plt.close()
        


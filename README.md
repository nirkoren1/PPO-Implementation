# PPO Implementation

<h1 align="center">
  <br>
  🤖 Proximal Policy Optimization (PPO) Implementation
  <br>
</h1>

<h4 align="center">A comprehensive PyTorch implementation of PPO with support for multiple environments and loss functions.</h4>

<p align="center">
  <a href="#key-features">Key Features</a> •
  <a href="#supported-environments">Supported Environments</a> •
  <a href="#installation">Installation</a> •
  <a href="#usage">Usage</a> •
  <a href="#project-structure">Project Structure</a> •
  <a href="#results">Results</a> •
  <a href="#visualization">Visualization</a>
</p>

<p align="center">
  <img src="walker2d.gif" alt="Walker2d PPO Agent Demo" width="600">
</p>

## Key Features

* **Multiple Loss Functions** - Implementation of different PPO variants:
  - **Clipped Loss** - Standard PPO with clipping (ε = 0.2)
  - **No Clip Loss** - Basic policy gradient without clipping
  - **KL Penalty Loss** - Adaptive KL divergence penalty method

* **Multi-Environment Support** - Works with:
  - **Classical Control** - CartPole, Pendulum environments
  - **MuJoCo Continuous Control** - Walker2d, HalfCheetah, Hopper, InvertedPendulum
  - **Atari Games** - Breakout, Alien, Jamesbond (with CNN state encoders)

* **Flexible Architecture**:
  - Configurable neural network architectures
  - Different state encoders for different environment types
  - Vectorized environments for parallel training
  - GAE (Generalized Advantage Estimation) support
  - Learning rate and clip coefficient annealing

* **Training Features**:
  - TensorBoard logging integration
  - Model checkpointing (saves best performing models)
  - Comprehensive experiment tracking
  - Parallel environment execution

## Supported Environments

### MuJoCo Continuous Control
- Walker2d-v5
- HalfCheetah-v5 
- Hopper-v5
- InvertedDoublePendulum-v5
- InvertedPendulum-v5
- Reacher-v5
- Swimmer-v5

### Atari Games
- ALE/Breakout-v5
- ALE/Alien-v5
- ALE/Jamesbond-v5

### Classical Control
- CartPole-v0/v1

## Installation

1. Clone this repository:
```bash
git clone https://github.com/nirkoren1/PPO-Implementation.git
cd PPO-Implementation
```

2. Install dependencies:
```bash
python -m venv myenv
.\myenv\Scripts\activate
pip install -r requirements.txt
```

## Usage

### Basic Training

Train PPO on a MuJoCo environment:
```bash
python code/ppo_train.py --env-name "Walker2d-v5" --state-encoder "NoEncoder" --total-timesteps 10000000 --num-steps 2048 --update-epochs 10 --minibatch-size 64 --adam-stepsize 3e-4 --gamma 0.99 --gae-lambda 0.95 --num-envs 1
```

Train on Atari with appropriate settings:
```bash
python code/ppo_train.py --env-name "ALE/Alien-v5" --state-encoder "AtariImageEncoder" --discrete-action --total-timesteps 10000000  --num-envs 8 --num-steps 128 --update-epochs 3 --minibatch-size 256 --adam-stepsize 2.5e-4 --gamma 0.99 --gae-lambda 0.95 --clip-coef 0.1 --vf-coef 1.0  --ent-coef 0.01 --anneal-lr --anneal-clip  --atari    
```


### Key Parameters

- `--env-name`: Environment name (e.g., 'Walker2d-v5', 'ALE/Breakout-v5')
- `--loss-type`: Loss function type ('clip', 'noclip', 'kl_penalty')
- `--clip-coef`: Clipping coefficient for PPO-Clip (default: 0.2)
- `--num-envs`: Number of parallel environments (default: 8)
- `--total-timesteps`: Total training timesteps (default: 10,000,000)
- `--state-encoder`: State encoder type ('NoEncoder', 'AtariImageEncoder', 'AtariNoEncoder')
- `--discrete-action`: Use discrete action space (required for Atari)
- `--atari`: Enable Atari-specific preprocessing

## Project Structure

```
PPO-Implementation/
├── code/
│   ├── ppo_train.py          # Main training script
│   ├── agent.py              # Actor-Critic neural networks
│   ├── losses.py             # PPO loss function implementations
│   ├── env_utils.py          # Environment wrappers
│   ├── replay_buffer.py      # Experience replay buffer
│   ├── state_encoder.py      # State preprocessing encoders
│   ├── constants.py          # Environment constants
│   ├── visualize.py          # Agent visualization script
│   ├── visualize_atari.py    # Atari-specific visualization
│   ├── experiments.ipynb     # Experiment notebooks
│   ├── models/               # Saved trained models
│   └── runs/                 # TensorBoard logs
├── requirements.txt          # Python dependencies
└── README.md                # This file
```

## Results

The implementation includes pre-trained models for various environments, demonstrating successful learning across different domains:

### Trained Models Available
- **MuJoCo Environments**: Walker2d, HalfCheetah, Hopper, InvertedPendulum variants
- **Classical Control**: CartPole

## Visualization

### View Trained Agents
```bash
python code/visualize.py --env-name Walker2d-v5 --model-path code/models/best_actor_Clip_Loss_Walker2d-v5.pth
```

### Atari Visualization
```bash
python code/visualize_atari.py --env-name ALE/Alien-v5 --model-path code/models/best_actor_Clip_Loss_ALE_Alien-v5.pth
```

## Experiments

The repository includes Jupyter notebooks with comprehensive experiments comparing different PPO variants across multiple environments. See:
- `code/experiments.ipynb` - MuJoCo environment experiments
- `code/commands.ipynb` - Example training commands

## Implementation Details

- **Algorithm**: PPO (Proximal Policy Optimization) with GAE
- **Networks**: Separate Actor-Critic with configurable hidden layers
- **Optimization**: Adam optimizer with optional learning rate annealing
- **Environments**: Vectorized parallel environments using Gymnasium
- **State Processing**: Environment-specific encoders: CNN for Atari, MLP for control

---
# Swarm Sim ML

Multi-agent reinforcement learning for cooperative drone swarm tactics. A swarm of drones learns to approach and hit a target from diverse angles using PPO with parameter sharing.

## Architecture

- **`drone.py`** — 3D point-mass drone model (double-integrator dynamics, drag, speed/accel limits)
- **`env.py`** — PettingZoo `ParallelEnv` with 8 drones, 39D observations, 3D acceleration actions
- **`train.py`** — PPO training via stable-baselines3 + SuperSuit, with eval entry point
- **`viewer.py`** — OpenGL FPV viewer with mouse look, WASD movement, drone trails, and HUD

## Setup

```
pip install -r requirements.txt
```

## Train

```
python train.py train --total-timesteps 2000000
```

Options: `--n-drones`, `--num-envs`, `--lr`, `--batch-size`, `--device`

Monitor training:

```
tensorboard --logdir ./logs/
```

## Evaluate

```
python train.py eval --model-path checkpoints/ppo_swarm_final
```

Opens the full 3D viewer with model-driven drones. Mouse to look, WASD to move, ESC to quit.

## Standalone Viewer

```
python viewer.py
```

Runs the viewer with random-walk drones (no trained model needed).

## How It Works

All drones share a single MLP policy network (parameter sharing via SuperSuit). Each drone observes its own position/velocity, the relative target vector, and the positions/velocities of its 5 nearest neighbors.

**Reward signal:**
- Approach shaping — reward for closing distance to target
- Angular spread — reward for approaching from a different angle than other drones
- Hit bonus — large reward on target contact, with bonus for unique approach angle
- Penalties — collisions, ground crashes, out-of-bounds, energy usage

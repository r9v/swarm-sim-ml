# Commands

## Train from scratch

```
python train.py train --total-timesteps 2000000
```

### Options

| Flag                | Default | Description                    |
| ------------------- | ------- | ------------------------------ |
| `--total-timesteps` | 2000000 | Total training steps           |
| `--n-drones`        | 8       | Number of drones               |
| `--num-envs`        | 4       | Parallel environments          |
| `--lr`              | 3e-4    | Learning rate                  |
| `--batch-size`      | 256     | Batch size                     |
| `--device`          | cpu     | Device (cpu/cuda)              |
| `--resume`          | None    | Checkpoint path to resume from |

## Resume training

```
python train.py train --resume checkpoints/ppo_swarm_final --total-timesteps 1000000
```

## Evaluate

```
python train.py eval --model-path checkpoints/ppo_swarm_final
```

### Options

| Flag           | Default    | Description         |
| -------------- | ---------- | ------------------- |
| `--model-path` | (required) | Path to saved model |
| `--n-drones`   | 8          | Number of drones    |

## TensorBoard

```
tensorboard --logdir ./logs/
```

import argparse
import os
import supersuit as ss
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import VecMonitor
from env import SwarmTargetEnv


def make_env(n_drones=8, render_mode=None):
    env = SwarmTargetEnv(n_drones=n_drones, render_mode=render_mode)
    env = ss.black_death_v3(env)
    env = ss.pad_observations_v0(env)
    env = ss.pad_action_space_v0(env)
    return env


def train(args):
    env = make_env(n_drones=args.n_drones)
    env = ss.pettingzoo_env_to_vec_env_v1(env)
    env = ss.concat_vec_envs_v1(env, num_vec_envs=args.num_envs, num_cpus=1, base_class="stable_baselines3")
    env = VecMonitor(env)

    os.makedirs("checkpoints", exist_ok=True)
    checkpoint_cb = CheckpointCallback(
        save_freq=max(args.total_timesteps // 10, 1000),
        save_path="checkpoints/",
        name_prefix="ppo_swarm",
    )

    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=args.lr,
        batch_size=args.batch_size,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        ent_coef=0.01,
        verbose=1,
        tensorboard_log="./logs/",
        device=args.device,
    )

    print(f"Training PPO | {args.total_timesteps} timesteps | {args.n_drones} drones | {args.num_envs} parallel envs")
    model.learn(total_timesteps=args.total_timesteps, callback=checkpoint_cb)
    model.save("checkpoints/ppo_swarm_final")
    env.close()
    print("Done. Model saved to checkpoints/ppo_swarm_final")
    print("View curves: tensorboard --logdir ./logs/")


def evaluate(args):
    from viewer import eval_loop
    env = SwarmTargetEnv(n_drones=args.n_drones)
    model = PPO.load(args.model_path, device="cpu")
    eval_loop(env, model)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)

    t = sub.add_parser("train")
    t.add_argument("--total-timesteps", type=int, default=2_000_000)
    t.add_argument("--n-drones", type=int, default=8)
    t.add_argument("--num-envs", type=int, default=4)
    t.add_argument("--lr", type=float, default=3e-4)
    t.add_argument("--batch-size", type=int, default=256)
    t.add_argument("--device", default="cpu")

    e = sub.add_parser("eval")
    e.add_argument("--model-path", required=True)
    e.add_argument("--n-drones", type=int, default=8)

    args = parser.parse_args()
    if args.command == "train":
        train(args)
    elif args.command == "eval":
        evaluate(args)

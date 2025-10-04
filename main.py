import os
os.environ['MUJOCO_GL'] = 'egl'
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import argparse
import torch
import numpy as np
import random

from replay_buffer import ReplayBuffer
from SAC.sac import SAC
from wrappers import make_env
from train import training
import shimmy

def main():
    parser = argparse.ArgumentParser(description='Soft Actor-Critic (SAC)')
    parser.add_argument('--env_name', default="quadruped-run",
                    help='DM control environment (default: hopper-hop).')
    parser.add_argument('--seed', type=int, default=0, help='random seed (default: 0).')
    parser.add_argument('--policy', default="Deterministic",
                    help='Policy Type: Gaussian | Deterministic (default: Gaussian).')
    parser.add_argument('--eval_ep', type=int, default=10,
            help='Number of episodes used for evaluation (default: 10).')
    parser.add_argument('--eval_interval', type=int, default=5000,
            help='Evaluates the policy every N episodes (default: 5000).')
    parser.add_argument('--automatic_entropy_tuning', type=bool, default=True,
                    help='Automaically adjust α default: True.')
    parser.add_argument('--alpha', type=float, default=0.2,
                    help='Temperature parameter α determines the relative importance of the entroy\
                            term against the reward (default: 0.2)')
    parser.add_argument('--max_steps', type=int, default=int(2e6),
            help='Number of training steps (default: 2e6).')
    parser.add_argument('--start_training', type=int, default=int(1e4),
                    help='Number of training steps to start training (default: 1e4).')
    parser.add_argument('--replay_size', type=int, default=int(1e6),
                    help='size of replay buffer (default: 1e6).')
    parser.add_argument('--batch_size', type=int, default=256,
                    help='Batch size (default: 256).')
    parser.add_argument('--lr', type=float, default=0.0003,
                    help='Learning rate (default: 0.0003).')
    parser.add_argument('--gamma', type=float, default=0.99,
                    help='Discount factor for reward (default: 0.99).')
    parser.add_argument('--tau', type=float, default=0.005,
                    help='Target smoothing coefficient (default: 0.005).')
    parser.add_argument('--hidden_size', type=int, default=256,
                    help='Hidden size (default: 256).')
    parser.add_argument('--updates_per_step', type=int, default=1,
                    help='Model updates per env step (default: 1).')
    parser.add_argument('--target_update_interval', type=int, default=1,
                    help='Value target update per no. of updates per step (default: 1).')
    parser.add_argument('--save_video', type=bool, default=False,
                    help='Save N=eval_ep videos during evaluation: False).')
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if device == 'cuda':
        torch.cuda.manual_seed_all(args.seed)

    args.save_dir = args.env_name+'/out/'
    if args.save_video:
        video_train_folder = args.env_name+'/video/'
    else:
        video_train_folder = None

    env = make_env(args.env_name, args.seed, None)
    env_eval = make_env(args.env_name, args.seed+42, video_train_folder)
    

    replay_buffer = ReplayBuffer(env.observation_space.shape, env.action_space.shape, args.replay_size, args.batch_size, device)
    sac = SAC(env.observation_space.shape[0], env.action_space.shape[0], env.action_space, device, args)

    training(env, env_eval, sac, replay_buffer, args, device)

if __name__ == '__main__':
    main()

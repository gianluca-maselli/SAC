# SAC (Soft Actor-Critic)
The repository contains the PyTorch implementation of the Soft Actor-Critic (SAC) described in ["Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor"](https://arxiv.org/pdf/1801.01290).

This implementation of the Soft Actor-Critic algorithm has been updated to support modern libraries and environments. It is compatible with Gymnasium and works with the DeepMind Control Suite.

## Resuts and Comparison
SAC was benchmarked on six environments from the DeepMind Control Suite. We evaluated a Gaussian policy (with automatic entropy tuning) against a discrete policy across 10 random seeds, reporting results with 95th percentile confidence intervals. Each experiment was run for 2e6 steps, and the same set of hyperparameters was used across all tasks for consistency.

<img src="images/results.png" width="1024"/>

## Requirements
Library  | Version
:-------------------------:|:-------------------------:
pytorch |  2.8.0
gymnasium | 1.2.0
mujoco | 3.3.5
dm_control | 1.0.31
numpy | 2.3.2
wheel | 0.45.1

## Usage
```
usage: main.py  [-h] [--env_name ENV_NAME] [--seed SEED] [--policy POLICY]
                [--eval_ep EVAL_EP] [--eval_interval EVAL_INTERVAL]
                [--automatic_entropy_tuning AUTOMATIC_ENTROPY_TUNING]
                [--alpha ALPHA] [--max_steps MAX_STEPS] [--start_training START_TRAINING]
                [--replay_size REPLAY_SIZE] [--batch_size BATCH_SIZE] [--lr LR]
                [--gamma GAMMA] [--tau TAU] [--hidden_size HIDDEN_SIZE]
                [--updates_per_step UPDATES_PER_STEP]
                [--target_update_interval TARGET_UPDATE_INTERVAL]
                [--save_video SAVE_VIDEO]

optional arguments:
  -h, --help                                               show this help message and exit
  --env_name ENV_NAME                                      DM control environment (default: hopper-hop).
  --seed SEED                                              random seed (default: 0).
  --policy POLICY                                          Policy Type: Gaussian | Deterministic (default: Gaussian).
  --eval_ep EVAL_EP                                        Number of episodes used for evaluation (default: 10).
  --eval_interval EVAL_INTERVAL                            Evaluates the policy every N episodes (default: 5000).
  --automatic_entropy_tuning AUTOMATIC_ENTROPY_TUNING      Automaically adjust α default: True.
  --alpha ALPHA                                            Temperature parameter α determines the relative importance of the entropy
                                                           term against the reward (default: 0.2)
  --max_steps MAX_STEPS                                    Number of training steps (default: 2e6).
  --start_training START_TRAINING                          Number of training steps to start training (default: 1e4).
  --replay_size REPLAY_SIZE                                size of replay buffer (default: 1e6).
  --batch_size BATCH_SIZE                                  Batch size (default: 256).
  --lr LR                                                  Learning rate (default: 0.0003).
  --gamma GAMMA                                            Discount factor for reward (default: 0.99).
  --tau TAU                                                Target smoothing coefficient (default: 0.005).
  --hidden_size HIDDEN_SIZE                                Hidden size (default: 256).
  --updates_per_step UPDATES_PER_STEP                      Model updates per env step (default: 1).
  --target_update_interval TARGET_UPDATE_INTERVAL          Value target update per no. of updates per step (default: 1).
  --save_video SAVE_VIDEO                                  Save N=eval_ep videos during evaluation: False).
```
## Acknowledgement
The code is inspired by the following implementation:
1. [pytorch-soft-actor-critic](https://github.com/pranz24/pytorch-soft-actor-critic) by [pranz24](https://github.com/pranz24)
2. [pytorch_sac](https://github.com/denisyarats/pytorch_sac) by [denisyarats](https://github.com/denisyarats)
3. [rl_with_resets](https://github.com/evgenii-nikishin/rl_with_resets) by [evgenii-nikishin](https://github.com/evgenii-nikishin)


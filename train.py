import numpy as np
import os
import tqdm

def evaluate(env, model, args):
    avg_reward = 0
    episodes = args.eval_ep

    for _ in range(episodes):
        obs, _ = env.reset()
        done = False
        truncated = False
        episode_reward = 0
        while not (done or truncated):
            action = model.select_action(obs, evaluate=True)
            next_obs, reward, done, truncated, _ = env.step(action)
            episode_reward += reward
            obs = next_obs
        avg_reward += episode_reward
    return avg_reward / episodes

def training(env, env_eval, sac, replay_buffer, args, device):
    #track stats
    tot_rew, updates = 0, 0
    eval_returns = []
    #init state
    obs, _  = env.reset()
    terminated = False
    truncated = False
    for step in tqdm.tqdm(range(1, args.max_steps + 1), smoothing=0.1):
        
        if step < args.start_training:
            #select random action
            action = env.action_space.sample()
        else:
            #select action accoring to the policy
            action = sac.select_action(obs)
        #step in the env
        next_obs, reward, terminated, truncated, info = env.step(action)
        tot_rew+=reward

        if not terminated or truncated: #'TimeLimit.truncated' in info:
            mask = 1
        else:
            mask = 0
        
        #add transition to replay buffer
        replay_buffer.add(obs, next_obs, action, reward, mask)

        obs = next_obs
        
        if terminated or truncated:
            tot_rew = 0
            obs, _ = env.reset()
            terminated = False
            truncated = False

        if len(replay_buffer) > args.batch_size:
            #update SAC critic and actor (update steps can be different from 1)
            for _ in range(args.updates_per_step):
                # Update parameters of all the networks
                critic_1_loss, critic_2_loss, policy_loss, ent_loss, alpha = sac.update(replay_buffer, updates)
                updates += 1

        if step % args.eval_interval == 0:
            test_avg_return = evaluate(env_eval, sac, args)
            #print(f'Test: {step}, Reward: {test_avg_return}')
            eval_returns.append((step,test_avg_return)) 
            np.savetxt(os.path.join(args.save_dir, f'{args.seed}.txt'), eval_returns, fmt=['%d', '%.1f'])

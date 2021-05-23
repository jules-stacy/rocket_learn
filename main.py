from rlgym.utils import reward_functions
from rlgym.utils.reward_functions import combined_reward
from rlgym.utils.terminal_conditions.common_conditions import GoalScoredCondition, TimeoutCondition
from rlgym.utils.reward_functions.common_rewards import TouchBallReward
from rlgym.utils.reward_functions.shoot_ball_reward import ShootBallReward
from rlgym.utils.reward_functions.combined_reward import CombinedReward
from cust_reward import GoalReward


import rlgym
# import time

import numpy as np
import torch
import argparse
import os

import agent_utils
import TD3
# import OurDDPG
# import DDPG


#setup name == main
if __name__ == "__main__":
    #add arguments to parser for future reference throughout main.py
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", default="TD3")                  # Policy name (TD3, DDPG or OurDDPG)
    parser.add_argument("--env", default="DuelSelf")                # RLgym environment name (Custom, Duel, Doubles, Standard, Basic)
    parser.add_argument("--opponents", default=True)                # Whether opponents are spawned or not
    parser.add_argument("--team_size", default=1)                   # Set team sizes
    parser.add_argument("--ep_len", default=None)                   # Set a default episode length in minutes
    parser.add_argument("--seed", default=0, type=int)              # Sets Gym, PyTorch and Numpy seeds
    parser.add_argument("--start_timesteps", default=25e3, type=int)# Time steps initial random policy is used
    parser.add_argument("--eval_freq", default=5e4, type=int)       # How often (time steps) we evaluate and save model
    parser.add_argument("--max_timesteps", default=1e6, type=int)   # Max time steps to run environment
    parser.add_argument("--expl_noise", default=0.1)                # Std of Gaussian exploration noise
    parser.add_argument("--batch_size", default=256, type=int)      # Batch size for both actor and critic
    parser.add_argument("--discount", default=0.99)                 # Discount factor
    parser.add_argument("--tau", default=0.005)                     # Target network update rate
    parser.add_argument("--policy_noise", default=0.2)              # Noise added to target policy during critic update
    parser.add_argument("--noise_clip", default=0.5)                # Range to clip target policy noise
    parser.add_argument("--policy_freq", default=2, type=int)       # Frequency of delayed policy updates
    parser.add_argument("--save_model", action="store_true")        # Save model and optimizer parameters
    parser.add_argument("--load_model", default="")                 # Model load file name, "" doesn't load, "default" uses file_name
    args = parser.parse_args() #add all the arguments to an args object

    file_name = f"{args.policy}_{args.env}_{args.seed}"
    print("---------------------------------------")
    print(f"Policy: {args.policy}, Env: {args.env}, Seed: {args.seed}")
    print("---------------------------------------")

    #create results storage and model storage
    if not os.path.exists("./results"):
        os.makedirs("./results")

    if args.save_model and not os.path.exists("./models"):
        os.makedirs("./models")


    # set episode timeout
    # 1800 sec = 30 min
    default_tick_skip = 8
    physics_ticks_per_second = 120
    ep_len_seconds = 1800
    max_steps = int(round(ep_len_seconds * physics_ticks_per_second / default_tick_skip))

    # specify terminal conditions from commons
    term_cond1 = TimeoutCondition(max_steps)
    term_cond2 = GoalScoredCondition()

    # specify reward functions
    
    goal_reward = ShootBallReward()
    touch_reward = TouchBallReward()

    all_rewards = CombinedReward(reward_functions=(goal_reward, touch_reward), reward_weights=(1.0,1.0))
    
    
    
    #create the environment
    env = rlgym.make(env_name=args.env, spawn_opponents=args.opponents, team_size=args.team_size, ep_len_minutes=args.ep_len, terminal_conditions=[term_cond2], reward_fn=all_rewards) #make the environment | 

    #set seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    #get dimensions of your obervation and action spaces
    #for further use in setting up your agents
    #Since DuelSelf will be active, action_space is going to be two controller states
    #thus we have to reference action_space[0] instead of just action_space
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0] 
    max_action = float(env.action_space.high[0])

    #set up your kwargs to feed into your agents
    kwargs = {
        "state_dim": obs_dim,
        "action_dim": action_dim,
        "max_action": max_action,
        "discount": args.discount,
        "tau": args.tau,
    }

    # Initialize policy
    #we're starting with TD3
    #we'll add additional policies later
    if args.policy == "TD3":
        # Target policy smoothing is scaled wrt the action scale
        kwargs["policy_noise"] = args.policy_noise * max_action
        kwargs["noise_clip"] = args.noise_clip * max_action
        kwargs["policy_freq"] = args.policy_freq
        policy = TD3.TD3(**kwargs)


    if args.load_model != "":
        policy_file = file_name if args.load_model == "default" else args.load_model
        policy.load(f"./models/{policy_file}")

    # initialize replay buffers for both agents
    replay_buffer = agent_utils.ReplayBuffer(obs_dim, action_dim)
    replay_buffer2 = agent_utils.ReplayBuffer(obs_dim, action_dim)

    
    episode_num = 0 #initialize the episode number
    t = 0
    # begin the training loop
    while True:
        # initialize observations
        obs, done = env.reset(), False
        obs1 = obs[0]
        obs2 = obs[1]
        # initialize rewards
        episode_reward1 = 0
        episode_reward2 = 0
        #initialize more stuff
        episode_timesteps = 0
        episode_num += 1 
        
        while not done:
            t += 1
            #add to the timesteps
            episode_timesteps += 1
            
            # Select action randomly or according to policy
            if t < args.start_timesteps:
                action1 = env.action_space.sample()
                action2 = env.action_space.sample()
            else:  # Select the action according to policy, with gaussian noise added
                action1 = (
                    policy.select_action(np.array(obs1))
                    + np.random.normal(0, max_action * args.expl_noise, size=action_dim)
                ).clip(-max_action, max_action) 

                action2 = (
                    policy.select_action(np.array(obs2))
                    + np.random.normal(0, max_action * args.expl_noise, size=action_dim)
                ).clip(-max_action, max_action)

            # store actions into a controller state list
            actions = [action1, action2]

            # Perform action
            new_obs, reward, done, state = env.step(actions)

            # Store data in replay buffers
            replay_buffer.add(obs1, action1, new_obs[0], reward[0], done)
            replay_buffer2.add(obs2, action2, new_obs[1], reward[1], done)

            # update observations
            obs1 = new_obs[0]
            obs2 = new_obs[1]

            episode_reward1 += reward[0]
            episode_reward2 += reward[1]

            # Train agent after collecting sufficient data
            if t >= args.start_timesteps:
                policy.train(replay_buffer, args.batch_size)

        print(f"Total T: {t+1} Episode Num: {episode_num+1} Episode T: {episode_timesteps} Blue Rwrd: {episode_reward1:.3f} Red Rwrd: {episode_reward2:.3f}")






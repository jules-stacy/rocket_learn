import rlgym
import time
from dqn_torch import Agent
import torch as T #base package

#rlPath = 'C:\\Program Files\\Epic Games\\rocketleague\\Binaries\\Win64\\RocketLeague.exe'


env = rlgym.make("Duel", spawn_opponents=True, team_size=1)
# print("the action space is: ", env.action_space.shape[0])
# print(env.observation_space)



#create the agent
agent = Agent(gamma = 0.99, epsilon = 1.0, batch_size = 64, n_actions = 8, eps_end = 0.01, input_dims = [30], lr = 0.003, action_space=[])

# device = T.device('cuda:0' if T.cuda.is_available else 'cpu')
# print(device)

# agent = RandomAgent()
# print(env.action_space.shape, env.observation_space.shape)

# #launch bakkesmod - single instance
# import subprocess
# bakkesmod = "C:\\Program Files\\BakkesMod\\BakkesMod.exe"
# subprocess.call([bakkesmod])

while True:
    observation = env.reset()
    done = False
    steps = 0
    ep_reward = 0
    t0 = time.time()
    while not done:
        action = agent.choose_action(observation)  # agent.act(obs) | Your agent should go here
        print(action)
        observation_, reward, done, state = env.step(action)
        ep_reward += reward
        agent.store_transition(observation, action, reward, observation_, done) #store to memory
        agent.learn() #call the learn function to update weights
        observation = observation_ #update the current observation
        steps += 1

    length = time.time() - t0
    print("Step time: {:1.5f} | Episode time: {:.2f} | Episode Reward: {:.2f}".format(length / steps, length, ep_reward))
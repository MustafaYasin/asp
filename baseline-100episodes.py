# -*- coding: utf-8 -*-

"""
This script used for evaluation the performance of different brains by playing against a baseline brain
"""

from unityagents import UnityEnvironment
import numpy as np
import torch
from algo.ddpg_agent import Agent

random_seed = 7

# Change the path to the env
env = UnityEnvironment(file_name="Tennis_Linux-0.4-Chen/Tennis_linux.x86_64", no_graphics=True)
# env = UnityEnvironment(file_name="Tennis_old.app")

# Change path to load baseline
agent_baseline = Agent(state_size=24, action_size=2, random_seed=random_seed)
brain_baseline = torch.load("a-c_7050-3.pth", map_location='cpu')
agent_baseline.actor_local.load_state_dict(brain_baseline['actor_static'])
agent_baseline.critic_local.load_state_dict(brain_baseline['critic_static'])

# Change path to load test brain
agent_test = Agent(state_size=24, action_size=2, random_seed=random_seed)
brain_test = torch.load("a-c_7900_j242912-3.pth", map_location='cpu')
agent_test.actor_local.load_state_dict(brain_baseline['actor_static'])
agent_test.critic_local.load_state_dict(brain_baseline['critic_static'])


brain_name = env.brain_names[0]
result_test = 0
episodes = 100
for i in range(episodes):
    env_info = env.reset(train_mode=False)[brain_name]     # reset the environment
    states = env_info.vector_observations                  # get the current state (for each agent)
    num_agents = len(env_info.agents)
    scores = np.zeros(num_agents)  # initialize the score (for each agent)
    while True:
        actions_baseline = agent_baseline.act(states)                        # select actions from loaded model agent
        actions_test = agent_test.act(states)
        actions = np.clip(np.array([actions_baseline[0], actions_test[1]]), -1, 1)       # all actions between -1 and 1
        env_info = env.step(actions)[brain_name]           # send all actions to tne environment
        next_states = env_info.vector_observations         # get next state (for each agent)
        rewards = env_info.rewards                         # get reward (for each agent)
        dones = env_info.local_done                        # see if episode finished
        scores += env_info.rewards                         # update the score (for each agent)
        states = next_states                               # roll over states to next time step
        if np.any(dones):                                  # exit loop if episode finished
            if scores[1] > scores[0]:
                result_test += 1
            break

print('Result of evaluation: the test agent wins {} of {} episodes'.format(result_test, episodes))
from unityagents import UnityEnvironment
import numpy as np
import torch
from algo.ddpg_agent import Agent

random_seed = 7

env = UnityEnvironment(file_name="Tennis_Linux/Tennis.x86_64")
# env = UnityEnvironment(file_name="Tennis_old.app")

agent = Agent(state_size=24, action_size=2, random_seed=random_seed)
agent.actor_local.load_state_dict(torch.load('checkpoint_actor_300.pth', map_location='cpu'))
agent.critic_local.load_state_dict(torch.load('checkpoint_critic_300.pth', map_location='cpu'))
brain_name = env.brain_names[0]
env_info = env.reset(train_mode=True)[brain_name]
num_agents = len(env_info.agents)
for i in range(100):                                         # play game for 5 episodes
    env_info = env.reset(train_mode=False)[brain_name]     # reset the environment
    states = env_info.vector_observations                  # get the current state (for each agent)
    scores = np.zeros(num_agents)                          # initialize the score (for each agent)
    while True:
        actions = agent.act(states)                        # select actions from loaded model agent
        actions = np.clip(actions, -1, 1)                  # all actions between -1 and 1
        env_info = env.step(actions)[brain_name]           # send all actions to tne environment
        next_states = env_info.vector_observations         # get next state (for each agent)
        rewards = env_info.rewards                         # get reward (for each agent)
        dones = env_info.local_done                        # see if episode finished
        scores += env_info.rewards                         # update the score (for each agent)
        states = next_states                               # roll over states to next time step
        if np.any(dones):                                  # exit loop if episode finished
            break
    print('Total score (averaged over agents) this episode: {}'.format(np.mean(scores)))

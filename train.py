import torch
import time
import numpy as np
import matplotlib.pyplot as plt

from datetime import datetime
from os import makedirs
from unityagents import UnityEnvironment
from algo.ddpg_agent import Agent
from collections import deque

# For linux
env = UnityEnvironment(file_name="Tennis_Linux/Tennis.x86_64", no_graphics=True)
# For mac os
# env = UnityEnvironment(file_name="Tennis_old.app")

# get the default brain
brain_name = env.brain_names[0]
brain = env.brains[brain_name]
# reset the environment
env_info = env.reset(train_mode=True)[brain_name]
# number of agents
num_agents = len(env_info.agents)
print('Number of agents:', num_agents)
# size of each action
action_size = brain.vector_action_space_size
print('Size of each action:', action_size)
# examine the state space
states = env_info.vector_observations
state_size = states.shape[1]
print('There are {} agents. Each observes a state with length: {}'.format(states.shape[0], state_size))
print('The state for the first agent looks like:', states[0])
random_seed = 7
train_mode = True

# Create one brain agent having one Reply memory buffer collecting experience from both tennis agents
agent = Agent(state_size=state_size, action_size=action_size, random_seed=random_seed)
save_dir = datetime.now().strftime("%m%d_%H:%M")
makedirs(save_dir, exist_ok=True)


def ddpg(n_episodes=2000, max_t=2000, print_every=5, save_every=50, learn_every=5, num_learn=10, goal_score=0.7):

  total_scores_deque = deque(maxlen=100)
  total_scores = []
  actor_losses = []
  critic_losses = []

  for i_episode in range(1, n_episodes + 1):

    # Reset Env and Agent
    env_info = env.reset(train_mode=train_mode)[brain_name]  # reset the environment
    states = env_info.vector_observations  # get the current state (for each agent)
    scores = np.zeros(num_agents)  # initialize the score (for each agent)

    agent.reset()
    
    start_time = time.time()

    for t in range(max_t):
      actions = agent.act(states)
      # print('actions:{}'.format(actions))
      env_info = env.step(actions)[brain_name]  # send all actions to the environment
      next_states = env_info.vector_observations  # get next state (for each agent)
      rewards = env_info.rewards  # get reward (for each agent)
      # print(rewards)
      dones = env_info.local_done  # see if episode finished
      # print('dones:{}'.format(dones))
      for state, action, reward, next_state, done in zip(states, actions, rewards, next_states, dones):
        agent.step(state, action, reward, next_state, done)  # send actions to the agent

      scores += env_info.rewards  # update the score (for each agent)
      states = next_states  # roll over states to next time step
      
      if t % learn_every == 0:
        for _ in range(num_learn):
          a_l, c_l = agent.start_learn()
          actor_losses.append(a_l * 1e3)
          critic_losses.append(c_l * 1e3)

      if np.any(dones):  # exit loop if episode finished
        break

    mean_score = np.mean(scores)
    min_score = np.min(scores)
    max_score = np.max(scores)

    # Take max of all agents' scores
    total_scores_deque.append(max_score)
    total_scores.append(max_score)
    total_average_score = np.mean(total_scores_deque)
    duration = time.time() - start_time


    if i_episode % print_every == 0:
      print(
        '\rEpisode {}\tTotal Average Score: {:.2f}\tMean: {:.2f}\tMin: {:.2f}\tMax: {:.2f}\tDuration: {:.2f}'.format(
          i_episode, total_average_score, mean_score, min_score, max_score, duration))

    if i_episode % save_every == 0:

      torch.save({'episode': i_episode,
                  'actor_static': agent.actor_local.state_dict(),
                  'actor_loss': actor_losses,
                  'critic_static': agent.critic_local.state_dict(),
                  'critic_loss': critic_losses,
                  'total_score': total_scores
                  }, '{}/a-c_{}.pth'.format(save_dir, i_episode))


    if total_average_score >= 0.8 and i_episode >= 100:
      print('Problem Solved after {} epsisodes!! Total Average score: {:.2f}'.format(i_episode, total_average_score))
      torch.save({'episode': i_episode,
                  'actor_static': agent.actor_local.state_dict(),
                  'actor_loss': actor_losses,
                  'critic_static': agent.critic_local.state_dict(),
                  'critic_loss': critic_losses,
                  'total_score': total_scores
                  }, '{}/a-c_{}.pth'.format(save_dir, i_episode))
      break

  return total_scores


scores = ddpg()
fig = plt.figure()
ax = fig.add_subplot(1,1,1)

plt.plot(np.arange(1, len(scores)+1), scores)
plt.ylabel('Average Score')
plt.xlabel('Episode No.')
plt.show()

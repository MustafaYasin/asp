# -*- coding: utf-8 -*-
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
env = UnityEnvironment(file_name="Tennis_Linux_0.4_final/Tennis_0.4_Final.x86_64", no_graphics=True)
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
continue_train = False
start_episodes = 1
if continue_train:
  trained = torch.load("a-c_2695-j244928-3.pth", map_location='cpu')
  agent.actor_local.load_state_dict(trained['actor_static'])
  agent.actor_target.load_state_dict(trained['actor_target'])
  agent.actor_local.eval()
  agent.actor_target.eval()
  agent.critic_local.load_state_dict(trained['critic_static'])
  agent.critic_target.load_state_dict(trained['critic_target'])
  agent.critic_local.eval()
  agent.critic_target.eval()
  start_episodes = trained['episode']

def ddpg(n_episodes=5000, max_t=2000, print_every=5, save_every=50, learn_every=10, goal_score=3, base_score=0.5):
  """

  :param n_episodes:
  :param max_t:
  :param print_every:
  :param save_every:
  :param learn_every: learn the
  :param num_learn:
  :param goal_score: the goal the model pursues
  :param base_score: stands for the baseline, after reaching which the model will save the train result
  :return:
  """
  total_scores_deque = deque(maxlen=100)
  total_scores = []
  actor_losses = []
  critic_losses = []

  for i_episode in range(start_episodes, start_episodes + n_episodes):

    # Reset Env and Agent
    env_info = env.reset(train_mode=train_mode)[brain_name]  # reset the environment
    states = env_info.vector_observations  # get the current state (for each agent)
    scores = np.zeros(num_agents)  # initialize the score (for each agent)

    agent.reset()
    start_time = time.time()

    for t in range(max_t):
      actions = agent.act(states)
      env_info = env.step(actions)[brain_name]  # send all actions to the environment
      next_states = env_info.vector_observations  # get next state (for each agent)
      rewards = env_info.rewards  # get reward (for each agent)
      dones = env_info.local_done  # see if episode finished
      for state, action, reward, next_state, done in zip(states, actions, rewards, next_states, dones):
        agent.step(state, action, reward, next_state, done)  # send actions to the agent

      scores += env_info.rewards  # update the score (for each agent)
      states = next_states  # roll over states to next time step
      
      if t % learn_every == 0:
        a_l, c_l = 0, 0
        for _ in range(learn_every):
          a_l, c_l = agent.start_learn()
        # potential memory leak
        actor_losses.append(a_l * 1e3)
        critic_losses.append(c_l * 1e3)
      if np.any(dones):  # exit loop if episode finished
        break

    max_score = np.max(scores)
    # Take max of all agents' scores
    total_scores_deque.append(max_score)
    total_scores.append(max_score)

    # mean total scores of last 100 episodes
    total_average_score = np.mean(total_scores_deque)
    duration = time.time() - start_time

    if i_episode % print_every == 0:
      print(
        '\rEpisode {}\tTotal Average Score: {:.2f}\tMax: {:.2f}\tDuration: {:.2f}'.format(
          i_episode, total_average_score, max_score, duration))

    if i_episode % save_every == 0 and i_episode <= 200:
      _torch_save(i_episode, agent, actor_losses, critic_losses, total_scores)

    if total_average_score > base_score and i_episode >= 1000:
      _torch_save(i_episode, agent, actor_losses, critic_losses, total_scores)
      base_score = total_average_score

    # check if goal archived
    if total_average_score >= goal_score and i_episode >= 1000:
        print('Problem Solved after {} episodes!! Total Average score: {:.2f}'.format(i_episode, total_average_score))
        _torch_save(i_episode, agent, actor_losses, critic_losses, total_scores)
        break

    # save the model at the end
  _torch_save(start_episodes + n_episodes, agent, actor_losses, critic_losses, total_scores)

  return total_scores

def _torch_save(episode, agent, actor_losses, critic_losses, total_scores):
  torch.save({'episode': episode,
              'actor_static': agent.actor_local.state_dict(),
              'actor_loss': actor_losses,
              'actor_target': agent.actor_target.state_dict(),
              'critic_static': agent.critic_local.state_dict(),
              'critic_loss': critic_losses,
              'critic_target': agent.critic_target.state_dict(),
              'total_score': total_scores
              }, '{}/a-c_{}.pth'.format(save_dir, episode))

scores = ddpg()
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)

plt.plot(np.arange(1, len(scores)+1), scores)
plt.ylabel('Average Score')
plt.xlabel('Episode No.')
plt.show()

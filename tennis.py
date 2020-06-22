from mlagents_envs.environment import UnityEnvironment
from asp.algo.ddpg_agent import Agent
import torch
import time
import numpy as np
from collections import deque

env = UnityEnvironment(file_name="tennis_demo.app", seed=1, side_channels=[])
random_seed = 10
train_mode = True

# Create one brain agent having one Reply memory buffer collecting experience from both tennis agents
agent = Agent(state_size=27, action_size=3, random_seed=random_seed)

def ddpg(n_episodes=2000, max_t=2000, print_every=5, save_every=50, learn_every=5, num_learn=10, goal_score=0.5):
  total_scores_deque = deque(maxlen=100)
  total_scores = []
  
  for i_episode in range(1, n_episodes + 1):
    # Reset Env and Agent
    env.reset()
    decision_steps, terminal_steps = env.get_steps(env.get_behavior_names()[0])
    states= decision_steps.obs[0]
    scores = np.zeros(2)  # initialize the score (for each agent)
    agent.reset()
    start_time = time.time()
    for t in range(max_t):
      if(len(list(terminal_steps.keys()))==0):actions = agent.act(states)

      print(f'decision obs:{decision_steps.obs}')
      env.set_actions(behavior_name=env.get_behavior_names()[0], action=actions)
      decision_steps, terminal_steps= env.get_steps(env.get_behavior_names()[0]) # get next state (for each agent)
      
      next_states = decision_steps.obs
      rewards = decision_steps.reward  # get reward (for each agent)
      
      dones = list(terminal_steps.keys())
      
      for state, action, reward, next_state, done in zip(states, actions, rewards, next_states, dones):
        agent.step(state, action, reward, next_state, done)  # send actions to the agent
      
      scores += decision_steps.reward  # update the score (for each agent)
      states = next_states  # roll over states to next time step
      
      if t % learn_every == 0:
        for _ in range(num_learn):
          agent.start_learn()
      
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
      print('\rEpisode {}\tTotal Average Score: {:.2f}\tMean: {:.2f}\tMin: {:.2f}\tMax: {:.2f}\tDuration: {:.2f}'.format(
        i_episode, total_average_score, mean_score, min_score, max_score, duration))
    
    if i_episode % save_every == 0:
      torch.save(agent.actor_local.state_dict(), 'checkpoint_actor.pth')
      torch.save(agent.critic_local.state_dict(), 'checkpoint_critic.pth')
    
    if total_average_score >= goal_score and i_episode >= 100:
      print('Problem Solved after {} epsisodes!! Total Average score: {:.2f}'.format(i_episode, total_average_score))
      torch.save(agent.actor_local.state_dict(), 'checkpoint_actor.pth')
      torch.save(agent.critic_local.state_dict(), 'checkpoint_critic.pth')
      break
    
    return total_scores
    
scores = ddpg()

# -*- coding: utf-8 -*-

import torch
import time
import numpy as np
import matplotlib.pyplot as plt
from mlagents_envs.environment import UnityEnvironment
from algo.ddpg_agent import Agent
from datetime import datetime
from os import makedirs
from collections import deque
env = UnityEnvironment(file_name="/home/Mao/workspace/ATP.ai/tennis_1_area/tennis-original.x86_64", seed=1,
                       side_channels=[], no_graphics=True)

random_seed = 10
# Create one brain agent having one Reply memory buffer collecting experience from both tennis agents
agent = Agent(state_size=27, action_size=3, random_seed=random_seed)
area_num = 1
agent_num = 2

save_dir = datetime.now().strftime("%m%d_%H:%M:%S")
makedirs(save_dir, exist_ok=True)
# scores
def check_done(steps):
    for s in steps:
        if len(s) > 0:
            return True
    return False


def ddpg(n_episodes=20000, max_t=2000, print_every=50, save_every=50, learn_every=5, num_learn=10):

    total_scores_deque = deque(maxlen=n_episodes)
    total_critic_losses = []
    total_actor_losses = []
    total_scores = []
    decision_steps = [0] * agent_num
    terminal_steps = [0] * agent_num
    actions = [0] * agent_num
    states = [0] * agent_num
    next_states = [0] * agent_num
    rewards = [0] * agent_num
    critic_loss, actor_loss = 0, 0
    for i_episode in range(1, n_episodes + 1):
        # Reset Env and Agent
        env.reset()
        for agent_id in range(agent_num):
            decision_steps[agent_id], terminal_steps[agent_id] = env.get_steps(env.get_behavior_names()[agent_id])
            states[agent_id] = decision_steps[agent_id].obs[0]

        # print(f'decision obs:{decision_steps.obs}')
        agent.reset()
        start_time = time.time()

        for t in range(max_t):

            states = np.array(states)
            actions = agent.act(states)

            for agent_id in range(agent_num):
                env.set_actions(behavior_name=env.get_behavior_names()[agent_id], action=actions[agent_id])
            env.step()

            for agent_id in range(agent_num):
                decision_steps[agent_id], terminal_steps[agent_id] = env.get_steps(env.get_behavior_names()[agent_id])

            done = check_done(terminal_steps)


            if not done:
                for agent_id in range(agent_num):
                    decision_steps[agent_id], terminal_steps[agent_id] = env.get_steps\
                        (env.get_behavior_names()[agent_id])
                    next_states[agent_id] = decision_steps[agent_id].obs[0]
                    rewards[agent_id] = decision_steps[agent_id].reward  # get reward (for each agent)

                    # scores += np.concatenate((decision_steps[].reward, decision_steps_1.reward), axis=0)
                    # update the score (for each agent)

            else:
                # scores += np.concatenate((terminal_steps_0.reward, terminal_steps_1.reward), axis=0)
                for agent_id in range(agent_num):
                    next_states[agent_id] = terminal_steps[agent_id].obs[0]
                    rewards[agent_id] = terminal_steps[agent_id].reward  # get reward (for each agent)


            for agent_id in range(agent_num):
                for state, action, reward, next_state in zip(states[agent_id], actions[agent_id], rewards[agent_id],
                                                             next_states[agent_id]):
                    agent.step(state, action, reward, next_state, done)  # send actions to the agent and save

            states = next_states  # roll over states to next time step

            if t % learn_every == 0:
                for _ in range(num_learn):
                    critic_loss, actor_loss = agent.start_learn()
            if done:
                break


        duration = time.time() - start_time
        total_scores.append(duration)
        total_critic_losses.append(critic_loss * 1e3)
        total_actor_losses.append(actor_loss * 1e3)



        print('Epoch {}: {}s'.format(i_episode, duration))

        if i_episode >= save_every:
            mean_score = np.mean(total_scores[-save_every:])
            min_score = np.min(total_scores[-save_every:])
            max_score = np.max(total_scores[-save_every:])

            if i_episode % print_every == 0:
                print(
                    '\rEpisode {}\tTotal Average Duration: {:.2f}s\tMean: {:.2f}s\t\tMin: {:.2f}s\tMax: {:.2f}s'.format(
                        i_episode, np.mean(total_scores), mean_score, min_score, max_score))


        if i_episode % save_every == 0:
            torch.save({'episode': i_episode,
                        'actor_static': agent.actor_local.state_dict(),
                        'actor_loss':total_actor_losses,
                        'critic_static': agent.critic_local.state_dict(),
                        'critic_loss':total_critic_losses,
                        'total_score': total_scores
                        }, '{}/a-c_{}.pth'.format(save_dir, i_episode))

        # if  i_episode >= 1000:
        #     # print('Problem Solved after {} episodes!! Total Average score: {:.2f}'.format(i_episode,
        #     #                                                                                total_average_score))
        #     torch.save(agent.actor_local.state_dict(), 'actor_old.pth')
        #     torch.save(agent.critic_local.state_dict(), 'checkpoint_critic_old.pth')
        #     break

    return total_scores, total_critic_losses, total_actor_losses


total_scores, total_critic_losses, total_actor_losses = ddpg()

plt.plot(total_scores, label='total_scores')
# plt.plot(test_losses_mean, label='test_loss_mean')
plt.plot(total_critic_losses, label='total_critic_losses')
# plt.plot(test_stress_losses_mean, label='test_stress_losses_mean')
plt.plot(total_actor_losses, label='total_actor_losses')
plt.legend(('total_scores',
            'total_critic_losses',
            'total_actor_losses'
            ),
          loc='upper right', shadow=True)
# plt.yscale("log")
plt.show()
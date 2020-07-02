from mlagents_envs.environment import UnityEnvironment
from algo.ddpg_agent import Agent
import torch
import time
import numpy as np
from collections import deque
env = UnityEnvironment(file_name="/home/m/mao/workspace/asp/tennis_1_area/tennis-original.x86_64", seed=1,
                       side_channels=[], no_graphics=True)

random_seed = 10
# Create one brain agent having one Reply memory buffer collecting experience from both tennis agents
agent = Agent(state_size=27, action_size=3, random_seed=random_seed)
area_num = 1
agent_num = 2

# scores
def check_done(steps):
    for s in steps:
        if len(s) > 0:
            return True
    return False


def ddpg(n_episodes=200000, max_t=2000, print_every=50, save_every=50, learn_every=5, num_learn=10):

    total_scores_deque = deque(maxlen=n_episodes)
    total_scores = []
    decision_steps = [0] * agent_num
    terminal_steps = [0] * agent_num
    actions = [0] * agent_num
    states = [0] * agent_num
    next_states = [0] * agent_num
    rewards = [0] * agent_num

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
            for agent_id in range(agent_num):

                actions[agent_id] = agent.act(states[agent_id])
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

                    # scores += np.concatenate((decision_steps[].reward, decision_steps_1.reward), axis=0)  # update the score (for each agent)

            else:
                # scores += np.concatenate((terminal_steps_0.reward, terminal_steps_1.reward), axis=0)
                for agent_id in range(agent_num):
                    next_states[agent_id] = terminal_steps[agent_id].obs[0]
                    rewards[agent_id] = terminal_steps[agent_id].reward  # get reward (for each agent)


            for agent_id in range(agent_num):
                for state, action, reward, next_state in zip(states[agent_id], actions[agent_id], rewards[agent_id], next_states[agent_id]):
                    agent.step(state, action, reward, next_state, done)  # send actions to the agent and save

            states = next_states  # roll over states to next time step

            if t % learn_every == 0:
                for _ in range(num_learn):
                    agent.start_learn()
            if done:
                break

        # TODO tensorboard加载， 每50个episode的平均时间

        # TODO slurm上不同hypopater

        # TODO evalution

        # Take max of all agents' scores
        # total_scores_deque.append(max_score)
        # total_scores.append(max_score)
        # total_average_score = np.mean(total_scores_deque)

        duration = time.time() - start_time
        total_scores.append(duration)


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
            torch.save(agent.actor_local.state_dict(), 'actor_{}.pth'.format(i_episode))
            torch.save(agent.critic_local.state_dict(), 'critic_{}.pth'.format(i_episode))

        # if  i_episode >= 1000:
        #     # print('Problem Solved after {} episodes!! Total Average score: {:.2f}'.format(i_episode,
        #     #                                                                                total_average_score))
        #     torch.save(agent.actor_local.state_dict(), 'actor_old.pth')
        #     torch.save(agent.critic_local.state_dict(), 'checkpoint_critic_old.pth')
        #     break

    return total_scores

ddpg()

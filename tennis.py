from mlagents_envs.environment import UnityEnvironment
from algo.ddpg_agent import Agent
import torch
import time
import numpy as np
from collections import deque

env = UnityEnvironment(file_name="/home/mustafa/Documents/unity/unity_env/unity_project_2_.x86_64", seed=1,
                       side_channels=[], no_graphics=True)
random_seed = 10
# Create one brain agent having one Reply memory buffer collecting experience from both tennis agents
agent = Agent(state_size=27, action_size=3, random_seed=random_seed)


def ddpg(n_episodes=2000, max_t=2000, print_every=5, save_every=50, learn_every=5, num_learn=10, goal_score=0.7):
    total_scores_deque = deque(maxlen=100)
    total_scores = []

    for i_episode in range(1, n_episodes + 1):
        # Reset Env and Agent
        env.reset()
        decision_steps_0, terminal_steps_0 = env.get_steps(env.behavior_specs()[0])
        decision_steps_1, terminal_steps_1 = env.get_steps(env.behavior_specs()[1])
        states_0 = decision_steps_0.obs[0]
        states_1 = decision_steps_1.obs[0]

        # print(f'decision obs:{decision_steps.obs}')
        scores = np.zeros(2)  # initialize the score (for each agent)
        agent.reset()
        start_time = time.time()
        for t in range(max_t):

            actions_0 = agent.act(states_0)
            actions_1 = agent.act(states_1)

            env.set_actions(behavior_name=env.get_behavior_names()[0], action=actions_0)
            env.set_actions(behavior_name=env.get_behavior_names()[1], action=actions_1)
            env.step()

            decision_steps_0, terminal_steps_0 = env.get_steps(
                env.get_behavior_names()[0])  # get next state (for each agent)
            decision_steps_1, terminal_steps_1 = env.get_steps(
                env.get_behavior_names()[1])
            # print(decision_steps_0)

            done = not (len(terminal_steps_0) == 0 & len(terminal_steps_1) == 0)

            if not done:
                next_states_0 = decision_steps_0.obs[0]
                next_states_1 = decision_steps_1.obs[0]

                rewards_0 = decision_steps_0.reward  # get reward (for each agent)
                rewards_1 = decision_steps_1.reward  # get reward (for each agent)
                scores += np.concatenate((decision_steps_0.reward, decision_steps_1.reward),
                                         axis=0)  # update the score (for each agent)
            else:
                scores += np.concatenate((terminal_steps_0.reward, terminal_steps_1.reward), axis=0)
                next_states_0 = terminal_steps_0.obs[0]
                next_states_1 = terminal_steps_1.obs[0]

                rewards_0 = terminal_steps_0.reward  # get reward (for each agent)
                rewards_1 = terminal_steps_1.reward  # get reward (for each agent)

                print(scores)

            for state, action, reward, next_state in zip(states_0, actions_0, rewards_0, next_states_0):
                agent.step(state, action, reward, next_state, done)  # send actions to the agent and save
            for state, action, reward, next_state in zip(states_1, actions_1, rewards_1, next_states_1):
                agent.step(state, action, reward, next_state, done)  # send actions to the agent and save

            states_0 = next_states_0  # roll over states to next time step
            states_1 = next_states_1  # roll over states to next time step

            if t % learn_every == 0:
                for _ in range(num_learn):
                    agent.start_learn()
            if done:
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
            torch.save(agent.actor_local.state_dict(), 'checkpoint_actor.pth')
            torch.save(agent.critic_local.state_dict(), 'checkpoint_critic.pth')

        if i_episode >= 1000:
            print('Problem Solved after {} epsisodes!! Total Average score: {:.2f}'.format(i_episode,
                                                                                           total_average_score))
            torch.save(agent.actor_local.state_dict(), 'checkpoint_actor.pth')
            torch.save(agent.critic_local.state_dict(), 'checkpoint_critic.pth')
            break

    return total_scores


ddpg()

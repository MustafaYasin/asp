# -*- coding: utf-8 -*-

# Load the saved weights into Pytorch model

from mlagents_envs.environment import UnityEnvironment
from algo.ddpg_agent import Agent
import torch
import numpy as np
random_seed = 10

agent = Agent(state_size=27, action_size=3, random_seed=random_seed)

trained = torch.load('/home/Mao/workspace/ATP.ai/06-13:49:17/a-c_200.pth', map_location='cpu')
agent.actor_local.load_state_dict(trained['actor_static'])
agent.critic_local.load_state_dict(trained['critic_static'])

env = UnityEnvironment(file_name="/home/Mao/workspace/ATP.ai/tennis_1_area/tennis-original.x86_64", seed=1,
                       side_channels=[], no_graphics=False)
area_num = 2

for i in range(100):

    env.reset()       # play game for 5 episodes
    for i in range(area_num):
        decision_steps_0, terminal_steps_0 = env.get_steps(env.get_behavior_names()[0])  # get next state (for each agent)
        decision_steps_1, terminal_steps_1 = env.get_steps(env.get_behavior_names()[1])

        states_0 = decision_steps_0.obs[0]
        states_1 = decision_steps_1.obs[0]

        scores = np.zeros(2)                          # initialize the score (for each agent)


        while True:

            actions_0 = np.clip(agent.act(states_0), -1, 1)
            actions_1 = np.clip(agent.act(states_1), -1, 1)


            env.set_actions(behavior_name=env.get_behavior_names()[0], action=actions_0)
            env.set_actions(behavior_name=env.get_behavior_names()[1], action=actions_1)

            env.step()   # send all actions to tne environment

            decision_steps_0, terminal_steps_0 = env.get_steps(
                env.get_behavior_names()[0])  # get next state (for each agent)
            decision_steps_1, terminal_steps_1 = env.get_steps(
                env.get_behavior_names()[1])

            done = not (len(terminal_steps_0) == 0 & len(terminal_steps_1) == 0)

            # see if episode finished
            if not done:
                next_states_0 = decision_steps_0.obs[0]
                next_states_1 = decision_steps_1.obs[0]

                rewards_0 = decision_steps_0.reward  # get reward (for each agent)
                rewards_1 = decision_steps_1.reward  # get reward (for each agent)
                scores += np.concatenate((decision_steps_0.reward, decision_steps_1.reward),
                                         axis=0)  # update the score (for each agent)

            else:
                next_states_0 = terminal_steps_0.obs[0]
                next_states_1 = terminal_steps_1.obs[0]

                rewards_0 = terminal_steps_0.reward  # get reward (for each agent)
                rewards_1 = terminal_steps_1.reward  # get reward (for each agent)
                scores += np.concatenate((rewards_0, rewards_1), axis=0)
                print(scores)
                break

            states_0 = next_states_0                               # roll over states to next time step
            states_1 = next_states_1  # roll over states to next time step


    print('Total score (averaged over agents) this episode: {}'.format(np.mean(scores)))


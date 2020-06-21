from mlagents_envs.environment import UnityEnvironment
import numpy as np
# This is a non-blocking call that only loads the environment.
# env = UnityEnvironment(file_name="tennis.x86_64", seed=1, side_channels=[])
# mac binary file
env = UnityEnvironment(file_name="tennis.app", seed=1, side_channels=[])
# env = UnityEnvironment(file_name="tennis_demo.app", seed=1, side_channels=[], )
# Start interacting with the evironment.
env.reset()
behavior_names = env.get_behavior_names()
brain_1 = env.get_behavior_names()[0]
# brain_2 = env.get_behavior_names()[1]

for behavior_name in behavior_names:
  behavior_specs=env.get_behavior_spec(behavior_name)
  print(str(behavior_name) + " " + str(behavior_specs))
  
for _ in range(1000):
  env.step()
  decision_steps, terminal_steps = env.get_steps(brain_1)
  print(f'decision_steps:{decision_steps.obs[0].shape}')
  print(list(terminal_steps.keys()))
  # if terminal_steps.keys
  if(len(list(terminal_steps.keys())) == 0):env.set_actions(behavior_name=brain_1, action=np.random.randn(18, 3))


env.close()



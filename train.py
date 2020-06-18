from mlagents_envs.environment import UnityEnvironment
# This is a non-blocking call that only loads the environment.
env = UnityEnvironment(file_name="tennis.x86_64", seed=1, side_channels=[])
# mac binary file
# env = UnityEnvironment(file_name="tennis.app", seed=1, side_channels=[])

# Start interacting with the evironment.
env.reset()
behavior_names = env.behavior_spec.keys()

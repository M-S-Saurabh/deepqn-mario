from stable_baselines3 import A2C
import gym_super_mario_bros
from wrappers import ScaledFloatFrame, MaxAndSkipEnv
from stable_baselines3.common.vec_env import VecFrameStack


# from stable_baselines3.common import make_vec_env
from gym_super_mario_bros.actions import RIGHT_ONLY
from nes_py.wrappers import JoypadSpace

# env = gym_super_mario_bros.make("SuperMarioBros-1-1-v0")
# env = JoypadSpace(env, RIGHT_ONLY)
# model = A2C('CnnPolicy', env, seed=0)
# model.learn(total_timesteps=500)
# model.save('a2c_mario')

trained_model = A2C.load("a2c_mario", verbose=1)

env = gym_super_mario_bros.make("SuperMarioBros-1-1-v0")
env = JoypadSpace(env, RIGHT_ONLY)
# env.num_envs = 1
# env = VecFrameStack(env, n_stack=4)

#---------
# instantiate your custom env
# obs = env.reset()
# # do 1000 random actions
# for i in range(1000):
#     env.render()
#     action = env.action_space.sample()
#     obs, reward, done, info = env.step(action)
#     if done:
#         obs = env.reset()
#---------

# from stable_baselines.common.env_checker import check_env
# check_env(env, warn=True)

trained_model.set_env(env)

env = gym_super_mario_bros.make("SuperMarioBros-1-1-v0")
obs = env.reset()
for i in range(2000):
    action, _states = trained_model.predict(obs)
    # print(action)
    obs, rewards, dones, info = env.step(action)
    env.render()
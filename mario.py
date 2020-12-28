'''
blog: https://blog.paperspace.com/building-double-deep-q-network-super-mario-bros/
source code: https://console.paperspace.com/gcn-team/notebook/pr5ddt1g9
'''

import random
import attr
from tqdm import tqdm
import pickle
import os

import torch
import torchviz
import gym_super_mario_bros

# User libraries
from wrappers import make_env
from networks import DQN
from agents import Memory, DQNAgent
from rewards import plot_rewards, save_rewards, load_rewards

@attr.s
class Config:
    batch_size = attr.ib(32)
    copy_step = attr.ib(5000)
    device = attr.ib("cuda")
    do_load_model = attr.ib(True)
    exploration_decay = attr.ib(0.99)
    exploration_max = attr.ib(1.0)
    exploration_min = attr.ib(0.02)
    gamma = attr.ib(0.90)
    is_training = attr.ib(False)
    learning_rate = attr.ib(0.00025)
    memory_size = attr.ib(30000)
    number_of_episodes = attr.ib(5000)

def run():
    config = Config()
    env = gym_super_mario_bros.make("SuperMarioBros-1-1-v0")
    env = make_env(env)

    state_shape = env.observation_space.shape
    output_size = env.action_space.n
    dqn = DQN(state_shape[0], config.batch_size, output_size)
    dqn_target = DQN(state_shape[0], config.batch_size, output_size)
    if config.is_training:
        memory = Memory(config.batch_size, config.memory_size, state_shape)
    else:
        memory = None

    agent = DQNAgent(dqn,
                       double_dqn = dqn_target,
                       copy_step=config.copy_step,
                       gamma=config.gamma,
                       lr=config.learning_rate,
                       exploration_max=config.exploration_max,
                       exploration_min=config.exploration_min,
                       exploration_decay=config.exploration_decay,
                       memory=memory,
                       device=config.device)

    max_reward = 0
    rewards = list()
    do_load = config.do_load_model or not config.is_training
    if do_load:
        agent.load()
        if config.is_training:
            agent.memory.load()
            rewards = load_rewards()

    if not config.is_training:
        agent.exploration_rate = 0.05

    for ep_num in tqdm(range(config.number_of_episodes)):
        state = env.reset()
        state = torch.from_numpy(state).unsqueeze(0)
        episode_reward = 0
        step = 0
        done = False
        while not done:
            env.render()
            action = agent.act(state)
            next_state, reward, done, info = env.step(action)
            if config.is_training:
                agent.q_update(state, action, reward, next_state, done)
                agent.update_exploration_rate()
            episode_reward += reward
            step += 1
            state = torch.from_numpy(next_state).unsqueeze(0)
            if done:
                break

        rewards.append(episode_reward)

        if config.is_training and (ep_num+1)%1000 == 0:
            agent.save()
            agent.memory.save()
            save_rewards(rewards)

    if config.is_training:
        agent.save()
        # agent.memory.save()
        save_rewards(rewards)
        plot_rewards(rewards)

from networks import ActorCriticNet
from agents import ActorCriticAgent

def runAC(loadpath=None, savepath='./saved_models/', params=None):
    config = Config()
    env = gym_super_mario_bros.make("SuperMarioBros-1-1-v0")
    env = make_env(env)

    max_episodes = 5000
    max_steps_per_episode = 1000

    if params is None:
        lr = 1e-4
        gamma = 0.99
        beta = 0.01
    else:
        lr = params['lr']
        gamma = params['gamma']
        beta = params['beta']

    state_shape = env.observation_space.shape
    output_size = env.action_space.n
    model = ActorCriticNet(state_shape[0], output_size)
    agent = ActorCriticAgent(model, lr=lr, gamma=gamma, beta=beta,
                    max_steps=max_steps_per_episode, device=config.device)

    if loadpath is None:
        running_reward = 0
        running_rewards = []
        episode_rewards = []
        episode_lengths = []
    else:
        agent.load(loadpath)
        running_rewards = load_rewards(loadpath+'running_rewards.pkl')
        episode_rewards = load_rewards(loadpath+'episode_rewards.pkl')
        episode_lengths = load_rewards(loadpath+'episode_lengths.pkl')
        running_reward = running_rewards[-1]

    with tqdm(range(max_episodes)) as t:
        for i in t:
            episode_reward, episode_length = agent.train_step(env)
            episode_reward = int(episode_reward)

            running_reward = episode_reward*0.01 + running_reward*.99

            running_rewards.append(running_reward)
            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)

            t.set_description(f'Episode {i}')
            t.set_postfix(episode_reward=episode_reward, running_reward=running_reward)
            
            if (i+1) % 100 == 0 : # or i == 0 
                agent.save(savepath)
                save_rewards(running_rewards, savepath+'running_rewards_lr{}_g{}_b{}.pkl'.format(params['lr'], params['gamma'], params['beta']))
                save_rewards(episode_rewards, savepath+'episode_rewards_lr{}_g{}_b{}.pkl'.format(params['lr'], params['gamma'], params['beta']))
                save_rewards(episode_lengths, savepath+'episode_lengths_lr{}_g{}_b{}.pkl'.format(params['lr'], params['gamma'], params['beta']))
    return

def testAC(loadpath='./saved_models/', params=None):
    config = Config()
    env = gym_super_mario_bros.make("SuperMarioBros-1-1-v0")
    env = make_env(env)

    max_episodes = 5000
    max_steps_per_episode = 1000

    if params is None:
        lr = 1e-4
        gamma = 0.99
        beta = 0.01
    else:
        lr = params['lr']
        gamma = params['gamma']
        beta = params['beta']

    state_shape = env.observation_space.shape
    output_size = env.action_space.n
    model = ActorCriticNet(state_shape[0], output_size)
    agent = ActorCriticAgent(model, lr=lr, gamma=gamma, beta=beta,
                    max_steps=max_steps_per_episode, device=config.device)
    agent.load(loadpath)

    if loadpath is None:
        running_reward = 0
        running_rewards = []
        episode_rewards = []
        episode_lengths = []
    else:
        agent.load(loadpath)
        running_rewards = load_rewards(loadpath+'running_rewards_lr{}_g{}_b{}.pkl'.format(params['lr'], params['gamma'], params['beta']))
        episode_rewards = load_rewards(loadpath+'episode_rewards_lr{}_g{}_b{}.pkl'.format(params['lr'], params['gamma'], params['beta']))
        episode_lengths = load_rewards(loadpath+'episode_lengths_lr{}_g{}_b{}.pkl'.format(params['lr'], params['gamma'], params['beta']))
        running_reward = running_rewards[-1]

    with tqdm(range(max_episodes)) as t:
        for i in t:
            states, actions, log_probs, values, rewards, last_Qval, num_steps = agent.run_episode(env, render=True)
            episode_reward = int(sum(rewards))

            t.set_description(f'Episode {i}')
            t.set_postfix(episode_reward=episode_reward, running_reward=running_reward)

def multiple_tests(prepath="/content/drive/MyDrive/8980-project-files/"):
    for lr in [1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2]:
        for gamma in [0.99, 0.9]:
            for beta in [1e-3, 5e-3, 1e-2, 1e-1]:
                params = {
                    'lr': lr,
                    'gamma': gamma,
                    'beta': beta
                }

                savepath = prepath+"A2C_trial_multiple_tests/"#.format(params['lr'], params['gamma'], params['beta'])

                if not os.path.exists(savepath):
                    os.makedirs(savepath)

                runAC(savepath = savepath)

if __name__ == "__main__":
    params = {
        'lr': 1e-5,
        'gamma': 0.95,
        'beta': 0.01
    }
    # runAC(savepath='./saved_models/actor_critic_test/', params=params)
    # multiple_tests("/content/drive/MyDrive/8980-project-files/")

    loadpath = './saved_models/A2C_trial_lr1e-05_g0.95_b0.01/'
    testAC(loadpath=loadpath, params=params)
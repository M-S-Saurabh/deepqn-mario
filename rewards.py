import matplotlib.pyplot as plt
import pickle

import numpy as np

def save_rewards(rewards, filename='rewards_list.pkl'):
    with open(filename, 'wb+') as temp:
        pickle.dump(rewards, temp)

def load_rewards(filename='rewards_list.pkl', loadpath=''):
    with open(loadpath+filename, 'rb') as temp:
        rewards = pickle.load(temp)
    return rewards

def smooth_rewards(rewards, n=10):
    rewards = np.array(rewards)
    smoothed = np.zeros_like(rewards)
    for i in range(n):
        smoothed[i] = rewards[i]
    for i in range(n-1, len(rewards)):
        smoothed[i] = np.mean(rewards[i-n+1:i+1])
    return smoothed

def plot_rewards(rewards, filename='rewards.png', params=None, savepath=''):
    plt.figure()
    plt.plot(rewards)
    plt.xlabel("Training Epochs")
    plt.ylabel("Cumulative reward over each epoch")
    if params is not None:
        plt.title("lr:{} g:{} b:{}".format(params['lr'], params['gamma'], params['beta']))
    plt.savefig(savepath+filename)
    
    # plt.show()
    plt.close()
    

def show_plots(prepath="/content/drive/MyDrive/8980-project-files/", plotname="episode_rewards"):
    loadpath = prepath+"A2C_trial_multiple_tests/"
    for lr in [1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2]:
        for gamma in [0.99, 0.9]:
            for beta in [1e-3, 5e-3, 1e-2, 1e-1]:
                params = {
                    'lr': lr,
                    'gamma': gamma,
                    'beta': beta
                }

                filename = loadpath+"{}_lr{}_g{}_b{}.pkl".format(
                                plotname, params['lr'], params['gamma'], params['beta'])

                rewards = load_rewards(filename)
                for i in range(len(rewards)-1, 0, -1):
                    if rewards[i] != 231:
                        print("I:", i, "reward:", rewards[i])
                        break
                plot_rewards(rewards, loadpath+'{}_lr{}_g{}_b{}.png'.format(
                                plotname, params['lr'], params['gamma'], params['beta']), params)

if __name__ == "__main__":
    loadpath = './saved_models/A2C_trial_lr1e-05_g0.95_b0.01/'
    rewards = load_rewards('episode_rewards_lr1e-05_g0.95_b0.01.pkl',loadpath=loadpath)
    plot_rewards(rewards, filename='rewards_actual.png', savepath=loadpath)
    plot_rewards(smooth_rewards(rewards, 20), filename='rewards_smoothed.png', savepath=loadpath)
    # show_plots(prepath=loadpath)

    # rewards = load_rewards(loadpath+'running_rewards.pkl')
    # plot_rewards(rewards, filename='running.png')

    # rewards = load_rewards(loadpath+'episode_lengths.pkl')
    # plot_rewards(rewards, filename='lengths.png')


import matplotlib.pyplot as plt
import pickle
from visdom import Visdom
import numpy as np

def save_rewards(rewards):
    with open('rewards_list.pkl', 'wb+') as temp:
        pickle.dump(rewards, temp)

def load_rewards():
    with open('rewards_list.pkl', 'rb') as temp:
        rewards = pickle.load(temp)
    return rewards

def plot_rewards(rewards):
    plt.figure()
    plt.plot(rewards)
    plt.xlabel("Training Epochs")
    plt.ylabel("Cumulative reward over each epoch")
    plt.savefig("rewards.png")
    plt.show()

class VisdomLinePlotter(object):
    """Plots to Visdom"""
    def __init__(self, env_name='main'):
        self.viz = Visdom()
        self.env = env_name
        self.plots = {}

    def plot(self, var_name, split_name, title_name, x, y):
        if var_name not in self.plots:
            self.plots[var_name] = self.viz.line(X=np.array([x,x]), Y=np.array([y,y]), env=self.env, opts=dict(
                legend=[split_name],
                title=title_name,
                xlabel='Epochs',
                ylabel=var_name
            ))
        else:
            self.viz.line(X=np.array([x]), Y=np.array([y]), env=self.env, win=self.plots[var_name], name=split_name, update = 'append')

if __name__ == "__main__":
    rewards = load_rewards()
    plot_rewards(rewards)
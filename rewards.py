import matplotlib.pyplot as plt
import pickle

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

if __name__ == "__main__":
    rewards = load_rewards()
    plot_rewards(rewards)
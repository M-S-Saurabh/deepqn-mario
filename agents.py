import random
import pickle
import torch
from torch import nn
import torchviz

from networks import DQN

class Memory:
    def __init__(self, memory_sample_size, memory_size, state_space):
        self.i = 0
        self.memory_sample_size = memory_sample_size
        self.memory_size = memory_size

        self.num_in_queue = 0
        self.actions = torch.zeros(memory_size, 1)
        self.dones = torch.zeros(memory_size, 1)
        self.next_states = torch.zeros(memory_size, *state_space)
        self.rewards = torch.zeros(memory_size, 1)
        self.states = torch.zeros(memory_size, *state_space)

    def remember(self, action, done, next_state, reward, state):
        i = self.i
        self.actions[i] = action.float()
        self.dones[i] = done.float()
        self.next_states[i] = next_state.float()
        self.rewards[i] = reward.float()
        self.states[i] = state.float()

        self.i = (i+1)%self.memory_size
        self.num_in_queue = min(self.num_in_queue+1, self.memory_size)

    def recall(self, device):
        idx = random.choices(range(self.num_in_queue), k=self.memory_sample_size)

        action = self.actions[idx].to(device)
        done = self.dones[idx].to(device)
        next_state = self.next_states[idx].to(device)
        reward = self.rewards[idx].to(device)
        state = self.states[idx].to(device)

        return action, done, next_state, reward, state
    
    def save(self):
        # Save class attributes
        with open('memory_dump.pkl', 'wb+') as memfile:
            pickle.dump( self.__dict__, memfile)

    def load(self):
        # Load class attributes
        with open('memory_dump.pkl', 'rb') as memfile:
            vars = pickle.load(memfile)
            for k,v in vars.items():
                setattr(self, k, v)
        
class DQNAgent:
    def __init__(self, dqn, gamma, lr, exploration_max, exploration_min, exploration_decay, memory, device="cpu", double_dqn=None, copy_step=None):
        self.Q = dqn
        self.Q.to(device)
        if double_dqn is not None:
            self.Q_target = double_dqn.to(device)
            self.copy_step = copy_step
        self.step = 0
        self.device = device
        self.exploration_decay = exploration_decay
        self.exploration_max = exploration_max
        self.exploration_min = exploration_min
        self.exploration_rate = exploration_max
        self.gamma = gamma
        self.loss_func = nn.SmoothL1Loss().to(self.device)#nn.MSELoss()
        self.memory = memory
        self.optimizer = torch.optim.Adam(self.Q.parameters(), lr=lr)

    def act(self, state):
        if random.random() < self.exploration_rate:
            action = random.randrange(self.Q.output_size)
        else:
            q = self.Q(state.to(self.device))
            action_tensor = torch.argmax(q)
            action = action_tensor.item()
        self.step += 1
        return action

    def q_update(self, _state, _action, _reward, _next_state, _done):
        if self.step % self.copy_step == 0:
            self.copy()
        _action = torch.tensor(_action)
        _reward = torch.tensor(_reward)
        _done = torch.tensor(1.0 if _done else 0)
        _next_state = torch.from_numpy(_next_state).unsqueeze(0)
        self.memory.remember(_action, _done, _next_state, _reward, _state)
        
        action, done, next_state, reward, state = self.memory.recall(self.device)
        self.optimizer.zero_grad()
        target_actions = self.Q_target(next_state).max(1).values.unsqueeze(1)
        target = reward + self.gamma*torch.mul(target_actions, 1 - done)
        current = self.Q(state).gather(1, action.long())
        loss = self.loss_func(current, target)
        loss.backward()
        self.optimizer.step()

    def copy(self):
        self.Q_target.load_state_dict(self.Q.state_dict())

    def save(self):
        torch.save(self.Q.state_dict(), "Q.pt")
        torch.save(self.Q_target.state_dict(), "Q_target.pt")

    def load(self):
        self.Q.load_state_dict(torch.load("Q.pt"))
        self.copy()

    def update_exploration_rate(self):
        r = self.exploration_decay*self.exploration_rate
        r = min(r, self.exploration_max)
        r = max(r, self.exploration_min)
        self.exploration_rate = r


from torch.distributions.categorical import Categorical
import torch.nn.functional as F
from networks import ActorCriticNet

class ActorCriticAgent:
    def __init__(self, nn_model, learning_rate, gamma=0.9, beta=0.01, max_steps=5000, device='cpu'):
        super().__init__()
        self.model = nn_model
        self.model.to(device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        self.max_steps = max_steps
        self.gamma = gamma
        self.beta = beta
        self.device = device
        
    def run_episode(self, env, nn_model, initial_state):
        state = initial_state
        log_prob_list = []
        value_list = []
        entropy_list = []
        reward_list = []
        for i in range(self.max_steps):
            # Run model and get action probs and critic values
            state = torch.from_numpy(state).unsqueeze(0).to(self.device)
            action_logits, value = nn_model(state)
            action_probs = F.softmax(action_logits, dim=1)
            log_probs = F.log_softmax(action_logits, dim=1)

            # Sample an action with these probs
            action = Categorical(action_probs).sample().item()

            # Run step with this action
            state, reward, done, _ = env.step(action)

            value_list.append(value)
            log_prob_list.append(log_probs[0, action])
            reward_list.append(reward)

            # Calculate entropy
            entropy_list.append( -(action_probs * log_probs).sum(1, keepdim=True) )

            if done: break
        return log_prob_list, value_list, reward_list, entropy_list

    def compute_loss(self, log_prob_list, value_list, reward_list, entropy_list):
        R = torch.zeros((1, 1), dtype=torch.float).to(self.device)
        # if not done:
        #     _, R, _, _ = local_model(state)

        gae = torch.zeros((1, 1), dtype=torch.float).to(self.device)
        actor_loss = 0
        critic_loss = 0
        entropy_loss = 0
        next_value = R

        for value, log_policy, reward, entropy in list(zip(value_list, log_prob_list, reward_list, entropy_list))[::-1]:
            gae = gae * self.gamma
            gae = gae + reward + self.gamma * next_value.detach() - value.detach()
            next_value = value
            actor_loss = actor_loss + log_policy * gae
            R = R * self.gamma + reward
            critic_loss = critic_loss + (R - value) ** 2 / 2
            entropy_loss = entropy_loss + entropy

        total_loss = -actor_loss + critic_loss - self.beta * entropy_loss
        return total_loss

    def train_step(self, env, model, initial_state):
        # Run a full episode
        log_prob_list, value_list, reward_list, entropy_list = self.run_episode(env, model, initial_state)

        # Compute loss
        total_loss = self.compute_loss(log_prob_list, value_list, reward_list, entropy_list)

        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

        return sum(reward_list)

    def save(self, path):
        torch.save(self.model.state_dict(), path+"model.pt")

    def load(self, path):
        self.model.load_state_dict(torch.load(path+"model.pt"))
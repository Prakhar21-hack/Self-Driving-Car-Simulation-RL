import numpy as np
import os
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

# Create the Architecture
class Network(nn.Module):

    def __init__(self, input_size, nb_action):
        super(Network, self).__init__()
        self.input_size = input_size
        self.nb_action = nb_action
        # Fully connected layer: each input neuron connected to hidden layer with 30 neurons
        self.fc1 = nn.Linear(input_size, 30)
        self.fc2 = nn.Linear(30, nb_action)

    # Forward propagation
    # Returns Q values for each possible action
    def forward(self, state):
        x = F.relu(self.fc1(state))
        q_values = self.fc2(x)
        return q_values

# Experience Replay
# Stores up to `capacity` experiences
class ReplayMemory(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []  # memory list

    def push(self, event):
        self.memory.append(event)
        if len(self.memory) > self.capacity:
            # Remove oldest element to maintain capacity
            del self.memory[0]

    def sample(self, batch_size):
        # Take a random sample of size batch_size from memory
        samples = zip(*random.sample(self.memory, batch_size))
        # Convert each sample to a torch Variable (tensor)
        return map(lambda x: Variable(torch.cat(x, 0)), samples)

# Deep Q Learning Agent
class Dqn():
    def __init__(self, input_size, nb_action, gamma):
        self.gamma = gamma
        self.reward_window = []
        self.model = Network(input_size, nb_action)
        self.memory = ReplayMemory(10000)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.last_state = torch.Tensor(input_size).unsqueeze(0)
        self.last_action = 0
        self.last_reward = 0

    def select_action(self, state):
        with torch.no_grad():
            # Note: specify dim=1 for softmax along action dimension
            probs = F.softmax(self.model(state) * 100, dim=1)
            # Sample 1 action from probability distribution
            action = probs.multinomial(num_samples=1)
        return action.item()

    def learn(self, batch_state, batch_next_state, batch_reward, batch_action):
        # Get predicted Q values for taken actions
        outputs = self.model(batch_state).gather(1, batch_action.unsqueeze(1)).squeeze(1)
        # Calculate max Q values for next states (detach so no gradient flow)
        next_outputs = self.model(batch_next_state).detach().max(1)[0]
        # Calculate target Q values
        target = self.gamma * next_outputs + batch_reward
        # Calculate loss (Huber loss)
        td_loss = F.smooth_l1_loss(outputs, target)
        self.optimizer.zero_grad()
        # PyTorch >= 0.4 uses retain_graph instead of retain_variables
        td_loss.backward(retain_graph=True)
        self.optimizer.step()

    def update(self, reward, new_signal):
        new_state = torch.Tensor(new_signal).float().unsqueeze(0)
        self.memory.push((self.last_state, new_state,
                          torch.LongTensor([int(self.last_action)]),
                          torch.Tensor([self.last_reward])))
        action = self.select_action(new_state)
        if len(self.memory.memory) > 100:
            batch_state, batch_next_state, batch_reward, batch_action = self.memory.sample(100)
            self.learn(batch_state, batch_next_state, batch_reward, batch_action)
        self.last_action = action
        self.last_state = new_state
        self.last_reward = reward
        self.reward_window.append(reward)
        if len(self.reward_window) > 1000:
            del self.reward_window[0]
        return action

    def score(self):
        return sum(self.reward_window) / (len(self.reward_window) + 1)

    def save(self):
        torch.save({
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict()
        }, 'last_brain.pth')

    def load(self):
        if os.path.isfile('last_brain.pth'):
            print('=> loading checkpoint...')
            checkpoint = torch.load('last_brain.pth')
            self.model.load_state_dict(checkpoint['state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            print('done!')
        else:
            print('No checkpoint found!')

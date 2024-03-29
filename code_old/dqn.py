# Date: 8/4/2023
# Author: Xubin Zhang
# Description: This file contains the implementation of...
import pandas as pd
import gymnasium as gym
import math
import random
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count

from environment import rp_env

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

env = rp_env()

# set up matplotlib
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display

plt.ion()

# if GPU is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Transition: A named tuple representing a single transition in an environment
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

#Use a cyclic buffer of bounded size that holds the transitions observed recently.
class ReplayMemory(object):
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

# Structure of DQN
class DQN(nn.Module):

    #Q-Network with 2 hidden layers, 128 neurons
    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(n_observations, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, n_actions)

    # Forward propagation with ReLU
    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)


##Training Phase

BATCH_SIZE = 128  # BATCH_SIZE is the number of transitions sampled from the replay buffer
GAMMA = 0.99  # GAMMA is the discount factor as mentioned in the previous section
EPS_START = 0.9  # EPS_START is the starting value of epsilon
EPS_END = 0.05  # EPS_END is the final value of epsilon
EPS_DECAY = 1000  # EPS_DECAY controls the rate of exponential decay of epsilon, higher means a slower decay
TAU = 0.005  # TAU is the update rate of the target network
LR = 1e-4  # LR is the learning rate of the ``AdamW`` optimizer

# Get number of actions from gym action space
n_actions = 22
# Get the number of state observations
state, info = env.reset()
n_observations = len(state)

policy_net = DQN(n_observations, n_actions).to(device)
target_net = DQN(n_observations, n_actions).to(device)
target_net.load_state_dict(policy_net.state_dict())

optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)
memory = ReplayMemory(10000)

steps_done = 0


# Select action by Epsilon-Greedy Policy according to state
def select_action(state):
    global steps_done
    sample = random.random()
    #Epsilon-Greedy Policy
    # Start with threshold=0.9,exploits most of the time with a small chance of exploring.
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1 # The value of threshold decreases, increasing the chance of exploration
    if sample > eps_threshold:
        # Exploitation, chooses the greedy action to get the most reward
        # by exploiting the agent’s current action-value estimates
        with torch.no_grad():
            # Use Q-network to calculate the max. Q-value
            # t.max(1) will return the largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            return policy_net(state).max(1)[1].view(1, 1)
    else:
        # Exploration, sample from the action space randomly
        return torch.tensor([[env.action_space.sample()]], device=device, dtype=torch.long)

episode_durations = [] # A list that keeps track of the duration of each episode for analysis after training is complete.

# Plot the duration of episodes, along with an average over the last 100 episodes.
# The plot will be underneath the cell containing the main training loop, and will update after every episode.
def plot_durations(show_result=False):
    plt.figure(1)
    durations_t = torch.tensor(episode_durations, dtype=torch.float)
    if show_result:
        plt.title('Result')
    else:
        plt.clf()
        plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.plot(durations_t.numpy())
    # Take 100 episode averages and plot them too
    if len(durations_t) >= 100:
        means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())

    plt.pause(0.001)  # pause a bit so that plots are updated
    if is_ipython:
        if not show_result:
            display.display(plt.gcf())
            display.clear_output(wait=True)
        else:
            display.display(plt.gcf())


# A single step of the optimization
def optimize_model():
    #Determine whether resampling is required
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)  # random sampling
    # First sample a batch of data from the memory, the size is BATCH_SIZE
    # Each data contains a state, an action, a reward and a next state
    # Unpack this tuple into four single lists, state, action, reward, and next state
    batch = Transition(*zip(*transitions))

    # Check states that in batch, create a boolean mask that identifies which states are non-final
    # Final state: False
    # Non-final state: Ture
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)), device=device, dtype=torch.bool)
    # Select next_state that is non-final state
    non_final_next_states = torch.cat([s for s in batch.next_state
                                                if s is not None])
    state_batch = torch.cat(batch.state)  # All states in batch
    action_batch = torch.cat(batch.action)  # All actions
    reward_batch = torch.cat(batch.reward)  # All rewards

    # Compute Q(s_t, a) by policy_net, then select the columns of actions taken.
    # These are the actions which would've been taken for each batch state according to policy_net
    state_action_values = policy_net(state_batch).gather(1, action_batch)

    # Compute V(s_{t+1}) for all next states by target-net.
    # Expected values of actions for non_final_next_states are computed based on the "older" target_net;
    # selecting their best reward with max(1)[0].
    # This is merged based on the mask, such that we'll have either the expected
    # state value or 0 in case the state was final.
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    with torch.no_grad(): # target_net
        next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0] # max. state-action value
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # Compute Huber loss
    # The Huber loss acts like the mean squared error when the error is small,
    # but like the mean absolute error when the error is large
    # This makes it more robust to outliers when the estimates of Q are very noisy.
    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

    # Optimize the model
    optimizer.zero_grad() #The gradient needs to be cleared before updating the parameters each time,
    # so as not to affect the next update due to the superposition of gradient information
    loss.backward() # Back-propagation: According to the previously calculated loss value,
    # the gradient is calculated by the chain rule, which is used to update the model parameters
    # In-place gradient clipping
    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100) #Clips gradient to ensure that the absolute value
    # of the gradient does not exceed 100, used to prevent the gradient explosion problem
    optimizer.step() # update weights

## Main Training Loop

if torch.cuda.is_available():
    num_episodes = 600
else:
    num_episodes = 50

for i_episode in range(num_episodes):
    # Initialize the environment and get it's state
    state, info = env.reset()
    state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)

    # Store the transition in csv file
    # Each episode has a file
    file_prefix = "transition_"
    filename = f"{file_prefix}{i_episode}.csv"
    columns = ["state", "action", "next_state", "reward"]

    try:
        df = pd.read_csv(filename)
    except FileNotFoundError:
        df = pd.DataFrame(columns=columns)

    for t in count():
        action = select_action(state)
        observation, reward, terminated, truncated, _ = env.step(action.item())
        reward = torch.tensor([reward], device=device)
        done = terminated or truncated

        if terminated == 1:
            next_state = None # Stop Episode
        else:
            next_state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)

        # Store the transition in memory
        memory.push(state, action, next_state, reward)

        # Store the transition in csv
        new_row = pd.Series([state, action, next_state, reward])
        df = df.append(new_row, ignore_index=True)

        # Move to the next state
        state = next_state

        # Perform one step of the optimization, just on the policy_net
        optimize_model()

        # Soft update of the target network's weights
        # θ′ ← τ θ + (1 −τ )θ′
        target_net_state_dict = target_net.state_dict()
        policy_net_state_dict = policy_net.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key]*TAU + target_net_state_dict[key]*(1-TAU)
        target_net.load_state_dict(target_net_state_dict)

        if done:
            episode_durations.append(t + 1)
            plot_durations()
            # Store all used transitions in an episode
            df.to_csv(filename, index=False)
            break

print('Complete')
plot_durations(show_result=True)
plt.ioff()
plt.show()


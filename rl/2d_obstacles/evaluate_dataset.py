"""
This function is used to test our environment dynamics propagation to make sure it can reproduce the dataset approximately
"""

import numpy as np
import time
import matplotlib.pyplot as plt
import torch
import argparse
import sys
import random
import seaborn as sns
from utils import *
from models.env_2d import obstacles, vertices
from pdb import set_trace as debug
from rl_utils.policies import get_expert_data, PolicyGaussian, PolicyAutoRegressiveModel
from env import Obstacle2DEnv

parser = argparse.ArgumentParser()
parser.add_argument('--simulate', action='store_true', default=False)
args = parser.parse_args()

# Get expert data
file_path = 'data/2d_obstacles_dataset.pkl'
expert_dataset = get_expert_data(file_path)
expert_data = expert_dataset["data"]
n_episodes = len(expert_data)
flattened_expert = {'observations': [], 
                    'actions': []}
for expert_path in expert_data:
    for k in flattened_expert.keys():
        flattened_expert[k].append(expert_path[k])
for k in flattened_expert.keys():
    flattened_expert[k] = np.concatenate(flattened_expert[k])

# Define environment
env = Obstacle2DEnv(expert_dataset["dt"], expert_dataset["pos_start_nom"], expert_dataset["pos_goal_nom"],
                    expert_dataset["rad_start_pos"], expert_dataset["rad_goal_pos"], expert_dataset["gcs_augment_degree"])

# Evaluate
states_episodes = []
actions_episodes = []
rewards_episodes = []
success_episodes = []
nx = expert_data[0]["observations"].shape[1]
nu = expert_data[0]["actions"].shape[1]
# Perform evaluation rollouts
for ep in range(n_episodes):
    if args.simulate:
        specifier = 'simulated'
        len_episode = expert_data[ep]["observations"].shape[0]
        actions = expert_data[ep]["actions"].T
        states = np.zeros((nx,len_episode))
        rewards = np.zeros(len_episode)
        env.reset()
        states[:,0] = env.cur_obs.reshape(-1)
        rewards[0] = env.evaluate_reward(states[:,0], actions[:,0])
        for ii in range(len_episode-1):
            state_next,reward,done = env.step(actions[:,ii].reshape(-1,1))
            states[:,ii+1] = state_next.reshape(-1)
            rewards[ii+1] = reward
    else:
        specifier = 'raw'
        states = expert_data[ep]["observations"].T
        actions = expert_data[ep]["actions"].T
        rewards = expert_data[ep]["rewards"].T
        done = env.evaluate_done(states[:,-1])
    states_episodes.append(states)
    actions_episodes.append(actions)
    rewards_episodes.append(rewards)
    success_episodes.append(done)
    print("episode: "+str(ep+1))

# == Plotting ==
setup_fig()
plot_environment(env)
for ep in range(n_episodes):
    alpha = 1. / n_episodes
    # color = 'green' if success_episodes[ep] else 'red'
    color = 'blue'
    plt.scatter(states_episodes[ep][0, 0], states_episodes[ep][1, 0], 2, alpha=min(3*alpha,1), color=color, zorder=5, linewidth=0, marker='x')
    plt.scatter(states_episodes[ep][0,-1], states_episodes[ep][1,-1], 2, alpha=min(3*alpha,1), color=color, zorder=5, linewidth=0, marker='o')
    plt.plot(states_episodes[ep][0,:], states_episodes[ep][1,:], color, alpha=alpha, linewidth=1, zorder=5)
plt.savefig('figures/dataset_'+specifier+'.pdf', bbox_inches='tight')
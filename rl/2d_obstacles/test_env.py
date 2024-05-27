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
sys.path.append("../utils/")

from policies import get_expert_data, PolicyGaussian, PolicyAutoRegressiveModel
from env import Obstacle2DEnv

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
                        expert_dataset["rad_start_pos"], expert_dataset["rad_goal_pos"])

# Perform manual rollouts
nx = expert_data[0]["observations"].shape[1]
nu = expert_data[0]["actions"].shape[1]
states_episodes = []
actions_episodes = []
success_episodes = []
for ep in range(n_episodes):
    len_episode = expert_data[ep]["observations"].shape[0]
    actions = expert_data[ep]["actions"].T
    states = np.zeros((nx,len_episode))
    env.reset()
    states[:,0] = env.cur_obs.reshape(-1)
    for ii in range(len_episode-1):
        state_next,_,_ = env.step(actions[:,ii].reshape(-1,1))
        states[:,ii+1] = state_next.reshape(-1)
    states_episodes.append(states)
    actions_episodes.append(actions)
    success_episodes.append(env.evaluate_done(state_next))
    print("Finished episode: "+str(ep+1))
    
# == Plotting ==
x_min = np.min(np.vstack(vertices), axis=0)
x_max = np.max(np.vstack(vertices), axis=0)
vertices = augment_gcs_vertices(vertices, degree=expert_dataset["gcs_augment_degree"])
setup_fig(x_min, x_max)
plot_environment(obstacles, vertices)
for ep in range(n_episodes):
    alpha = 0.1
    color = 'green' if success_episodes[ep] else 'red'
    plt.plot(states_episodes[ep][0,:], states_episodes[ep][1,:], color, alpha=alpha, linewidth=1, zorder=5)
plt.savefig('figures/env_test_rollout.pdf', bbox_inches='tight')
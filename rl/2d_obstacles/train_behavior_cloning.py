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

from policy_utils import get_expert_data, PolicyGaussian, PolicyAutoRegressiveModel
from evaluate import evaluate
from bc import simulate_policy_bc
from env import Obstacle2DEnv

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('using device', device)

torch.manual_seed(0)
random.seed(0)
np.random.seed(0)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--test', action='store_true', default=False)
    parser.add_argument('--render',  action='store_true', default=False)
    args = parser.parse_args()

    # Get the expert data
    file_path = 'data/2d_obstacles_dataset.pkl'
    expert_data = get_expert_data(file_path)
    n_episodes = len(expert_data)
    flattened_expert = {'observations': [], 
                        'actions': []}
    for expert_path in expert_data:
        for k in flattened_expert.keys():
            flattened_expert[k].append(expert_path[k])
    for k in flattened_expert.keys():
        flattened_expert[k] = np.concatenate(flattened_expert[k])

    # Define policy
    hidden_dim = 128
    hidden_depth = 2
    obs_size = flattened_expert['observations'][0].size
    ac_size = flattened_expert['actions'][0].size
    ac_margin = 0.1
    policy_type = 'autoregressive'
    if policy_type == 'gaussian':
        policy = PolicyGaussian(num_inputs=obs_size, num_outputs=ac_size, hidden_dim=hidden_dim, hidden_depth=hidden_depth)
    elif policy_type == 'autoregressive':
        num_buckets = 10
        policy = PolicyAutoRegressiveModel(num_inputs=obs_size, num_outputs=ac_size, hidden_dim=hidden_dim, 
                                            hidden_depth=hidden_depth, num_buckets=num_buckets, 
                                            ac_low=flattened_expert['actions'].min(axis=0) - ac_margin, 
                                            ac_high=flattened_expert['actions'].max(axis=0) + ac_margin)
    policy.to(device)

    # Define environment
    init_conds = np.array([expert_data[ep]["observations"][0,:] for ep in range(n_episodes)])
    term_conds = np.array([expert_data[ep]["observations"][-1,:] for ep in range(n_episodes)])
    dt = 0.1 # check generate_dataset.py for this hyperparameter
    env = Obstacle2DEnv(init_conds, term_conds, dt)

    # Training hyperparameters for BC
    episode_length = max([len(expert_data[ep]['observations']) for ep in range(n_episodes)])
    num_epochs = 500
    batch_size = 64

    # Train behavior cloning
    if not args.test:
        train_losses = simulate_policy_bc(policy, expert_data, num_epochs=num_epochs, episode_length=episode_length, batch_size=batch_size)
        torch.save(policy.state_dict(), f'policies/2d_obstacle_policy.pth')
    else:
        policy.load_state_dict(torch.load(f'policies/2d_obstacle_policy.pth'))
        
    # Test behavior cloning
    agent_name = 'bc'
    num_validation_runs = 100
    (evaluation_paths, success_rate, avg_return) = evaluate(env, policy, agent_name, num_validation_runs=num_validation_runs, episode_length=episode_length)
    
    # == Plotting ==
    # Trajectories plot
    savefig = True
    x_min = np.min(np.vstack(vertices), axis=0)
    x_max = np.max(np.vstack(vertices), axis=0)
    setup_fig(x_min, x_max)
    plot_environment(obstacles, vertices)
    for run in range(num_validation_runs):
        alpha = .1 # 10./num_validation_runs
        positions_run = evaluation_paths[run]["observations"][:,0:2]
        successful_run = any(list(evaluation_paths[run]["dones"]))
        color = 'green' if successful_run else 'red'
        plt.plot(positions_run[:,0], positions_run[:,1], color, alpha=alpha, linewidth=1, zorder=5)
    if savefig:
        plt.savefig('figures/policy_evaluation.pdf', bbox_inches='tight')
        
    # Loss plot
    sns.set_theme()
    fig = plt.figure()
    plt.plot(range(num_epochs), train_losses, color='r', linewidth=2)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.savefig("figures/loss.pdf")
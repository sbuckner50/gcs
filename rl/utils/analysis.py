import gym
import numpy as np
import time
import matplotlib.pyplot as plt
import torch
import argparse
from pdb import set_trace as debug

from gcs.rl.utils.policy_utils import generate_paths, get_expert_data, PolicyGaussian, PolicyAutoRegressiveModel
from bc import simulate_policy_bc
from dagger import simulate_policy_dagger
import pytorch_utils as ptu
from evaluate import evaluate
from reach_goal.envs.pointmaze_env import PointMazeEnv
from reach_goal.envs.pointmaze_expert import WaypointController

def train_and_evaluate(
    arg_env, 
    arg_train, 
    arg_policy, 
    test=False, 
    render=False, 
    device='cuda', 
    episode_length=None, 
    num_epochs=None, 
    batch_size=None, 
    num_dagger_iters=None, 
    num_trajs_per_dagger=None,
    num_buckets=None):   

    # Get the expert data
    if arg_env == 'reacher':
        file_path = 'data/reacher_expert_data.pkl'
    elif arg_env == 'pointmaze':
        file_path = 'data/pointmaze_expert_data.pkl'
    else:
        raise ValueError('Invalid environment')
    expert_data = get_expert_data(file_path)

    flattened_expert = {'observations': [], 
                        'actions': []}
    
    for expert_path in expert_data:
        for k in flattened_expert.keys():
            flattened_expert[k].append(expert_path[k])

    for k in flattened_expert.keys():
        flattened_expert[k] = np.concatenate(flattened_expert[k])
    
    # Define environment
    if arg_env == 'reacher':
        env = gym.make("Reacher-v2")
        
    elif arg_env == 'pointmaze':
        env = PointMazeEnv(render_mode='human' if render else 'rgb_array')
    else:
        raise ValueError('Invalid environment')

    # Define policy
    hidden_dim = 128
    hidden_depth = 2
    obs_size = env.observation_space.shape[0]
    ac_size = env.action_space.shape[0]
    ac_margin = 0.1

    if arg_policy == 'gaussian':
        policy = PolicyGaussian(num_inputs=obs_size, num_outputs=ac_size, hidden_dim=hidden_dim, hidden_depth=hidden_depth)
    elif arg_policy == 'autoregressive':
        num_buckets = 10 if num_buckets is None else num_buckets
        policy = PolicyAutoRegressiveModel(num_inputs=obs_size, num_outputs=ac_size, hidden_dim=hidden_dim, 
                                            hidden_depth=hidden_depth, num_buckets=num_buckets, 
                                            ac_low=flattened_expert['actions'].min(axis=0) - ac_margin, 
                                            ac_high=flattened_expert['actions'].max(axis=0) + ac_margin)
    policy.to(device)

    # Training hyperparameters for BC
    if arg_env == 'reacher':
        episode_length = 50 if episode_length is None else episode_length
        num_epochs = 500 if num_epochs is None else num_epochs
        batch_size = 32 if batch_size is None else batch_size
    elif arg_env == 'pointmaze':
        episode_length = 300 if episode_length is None else episode_length
        num_epochs = 10 if num_epochs is None else num_epochs
        batch_size = 128 if batch_size is None else batch_size
    else:
        raise ValueError('Invalid environment')

    train_losses = []
    if not test:
        if arg_train == 'behavior_cloning':
            # Train behavior cloning
            train_losses = simulate_policy_bc(env, policy, expert_data, num_epochs=num_epochs, episode_length=episode_length,
                            batch_size=batch_size)
        elif arg_train == 'dagger':
            if arg_env == 'reacher':
                # Load interactive expert
                expert_policy = torch.load('data/reacher_expert_policy.pkl', map_location=torch.device(device))
                print("Expert policy loaded")
                expert_policy.to(device)
                ptu.set_gpu_mode(True)
            elif arg_env == 'pointmaze':
                expert_policy = WaypointController(env.maze)
            else:
                raise ValueError('Invalid environment')

            # Training hyperparameters for DAgger
            num_dagger_iters=10 if num_dagger_iters is None else num_dagger_iters
            num_epochs = int(num_epochs/num_dagger_iters)
            num_trajs_per_dagger=10 if num_trajs_per_dagger is None else num_trajs_per_dagger
            # Train DAgger
            train_losses = simulate_policy_dagger(env, policy, expert_data, expert_policy, num_epochs=num_epochs, episode_length=episode_length,
                            batch_size=batch_size, num_dagger_iters=num_dagger_iters, num_trajs_per_dagger=num_trajs_per_dagger)
        else:
            raise ValueError('Invalid training method')
        # Code for saving a policy to a checkpoint
        torch.save(policy.state_dict(), f'{arg_policy}_{arg_env}_{arg_train}_final.pth')
    else:
        # Code for loading a policy from a checkpoint
        policy.load_state_dict(torch.load(f'{arg_policy}_{arg_env}_{arg_train}_final.pth'))

    # Code for policy evaluation post training
    (success_rate, avg_return) = evaluate(env, policy, arg_train, num_validation_runs=100, episode_length=episode_length, render=render, env_name=arg_env)
    
    return train_losses, success_rate, avg_return
    
def plot_compare_reacher_pointmaze(
    x_data_reacher,
    y_data_reacher,
    x_data_pointmaze,
    y_data_pointmaze,
    xlabel = 'Training Progress (%)',
    ylabel = 'Loss',
    twinx = True
    ):
    fig, ax1 = plt.subplots()
    if twinx:
        ax2 = ax1.twinx()
    else:
        ax2 = ax1
    color_reacher_dark = 'blue'
    color_reacher_light = 'skyblue'
    color_pointmaze_dark = 'red'
    color_pointmaze_light = 'lightsalmon'
    ax1.plot(x_data_reacher, y_data_reacher, color=color_reacher_dark, linewidth=.5, label='Reacher')
    ax1.scatter(x_data_reacher, y_data_reacher, edgecolor=color_reacher_dark, facecolor=color_reacher_light, s=6)
    ax2.plot(x_data_pointmaze, y_data_pointmaze, color=color_pointmaze_dark, linewidth=.5, label='Pointmaze')
    ax2.scatter(x_data_pointmaze, y_data_pointmaze, edgecolor=color_pointmaze_dark, facecolor=color_pointmaze_light, s=6)
    ax1.set_xlabel(xlabel)
    if twinx:
        ax1.set_ylabel('Reacher '+ylabel)
        ax2.set_ylabel('Pointmaze '+ylabel)
        ax1.spines['left'].set_color(color_reacher_dark)
        ax1.yaxis.label.set_color(color_reacher_dark)
        ax1.tick_params(axis='y', colors=color_reacher_dark)
        ax2.spines['right'].set_color(color_pointmaze_dark)
        ax2.yaxis.label.set_color(color_pointmaze_dark)
        ax2.tick_params(axis='y', colors=color_pointmaze_dark)
    else:
        ax1.set_ylabel(ylabel)
        ax1.legend()
    plt.show()
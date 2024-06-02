import numpy as np
import time
import matplotlib.pyplot as plt
import torch
import argparse
import sys
import random
import pickle
import seaborn as sns
from utils import *
from models.env_2d import obstacles, vertices
from pdb import set_trace as debug

from rl_utils.policies import get_expert_data, DAPGPolicy, PGBaseline
from rl_utils.dapg_algorithm import *
from rl_utils.evaluate import evaluate
from utils import *
from env import Obstacle2DEnv

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('using device', device)

torch.manual_seed(0)
random.seed(0)
np.random.seed(0)

# Comparison parameters
fnames = ('bc_v1','pg_v1','dapgweightdist')
labels = ('BC', 'PG', 'DAPG')
colors = ('red', 'blue', 'green')

# Other parameters
num_validation_runs = 50
# alpha_trajs = 1/num_validation_runs
alpha_trajs = 0.5

# Get the expert data
file_path = 'data/2d_obstacles_dataset.pkl'
expert_dataset = get_expert_data(file_path)
expert_data = expert_dataset["data"]
n_episodes = len(expert_data)
max_episode_length = expert_dataset["max_episode_length"]

# Define policy (see train_DAPG.py)
hidden_dim_pol = 64
hidden_depth_pol = 2
obs_size = expert_data[0]['observations'][0].size
ac_size = expert_data[0]['actions'][0].size
policy = DAPGPolicy(obs_size, ac_size, hidden_dim=hidden_dim_pol, hidden_depth=hidden_depth_pol)
policy.to(device)

# Define environment
env = Obstacle2DEnv(expert_dataset["dt"], expert_dataset["pos_start_nom"], expert_dataset["pos_goal_nom"],
                    expert_dataset["rad_start_pos"], expert_dataset["rad_goal_pos"], expert_dataset["gcs_augment_degree"])
    
# Define plots
sns.set_theme()
fig, (ax_rew, ax_traj) = plt.subplots(1,2, gridspec_kw={'width_ratios': [2, 1]})
fig.set_figwidth(15)
ax_rew.set_xlabel('Epoch')
ax_rew.set_ylabel('Reward')
setup_fig(ax=ax_traj)
plot_environment(env, ax=ax_traj)

savefig_name = ''
for k in range(len(fnames)):
    # Load data
    policy.load_state_dict(torch.load(f'policies/policy_'+fnames[k]+'.pth'))
    with open(f'results/results_'+fnames[k]+'.pth', 'rb') as f:
        results = pickle.load(f)
        
    # Evaluate policy
    (evaluation_paths, success_rate, avg_return) = evaluate(env, policy, 'bc', num_validation_runs=num_validation_runs, episode_length=max_episode_length)
    
    # Add to reward plot
    train_rewards = results["train_rewards"]
    ax_rew.plot(range(len(train_rewards)), train_rewards, color=colors[k], label=labels[k])
    
    # Add to traj plot
    for run in range(num_validation_runs):
        positions_run = evaluation_paths[run]["observations"][:,:2]
        successful_run = any(list(evaluation_paths[run]["dones"]))
        plt.plot(positions_run[:,0], positions_run[:,1], color=colors[k], alpha=alpha_trajs, linewidth=1, zorder=5)
        
    # Other
    savefig_name += '_' + fnames[k]

ax_rew.legend()
plt.savefig('figures/compare'+savefig_name+'.pdf', bbox_inches='tight')
"""
TODO: MODIFY TO FILL IN YOUR DAGGER IMPLEMENTATION
"""
import torch
import torch.optim as optim
from pdb import set_trace as debug
import numpy as np

from policy_utils import rollout, relabel_action
from bc import simulate_policy_bc

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def simulate_policy_dagger(env, policy, expert_paths, expert_policy=None, num_epochs=500, episode_length=50,
                            batch_size=32, num_dagger_iters=10, num_trajs_per_dagger=10):

    # Dagger iterations
    losses = []
    returns = []
    trajs = expert_paths
    for dagger_itr in range(num_dagger_iters):
        
        # Train policy on current dataset (use simulate_policy_bc to reduce code duplication)
        losses += simulate_policy_bc(env, policy, trajs, num_epochs=num_epochs, episode_length=episode_length,
                                     batch_size=batch_size, shuffle_method='complete')

        # Collecting more data for dagger
        trajs_recent = []
        for _ in range(num_trajs_per_dagger):
            env.reset()
            traj = rollout(
                env,
                policy,
                agent_name='dagger',
                episode_length=episode_length)
            traj = relabel_action(traj, expert_policy)
            trajs_recent += [traj]

        trajs += trajs_recent
        mean_return = np.mean(np.array([traj['rewards'].sum() for traj in trajs_recent]))
        print("Average DAgger return is " + str(mean_return))
        returns.append(mean_return)

    return losses
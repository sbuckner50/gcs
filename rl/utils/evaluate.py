"""
DO NOT MODIFY BESIDES HYPERPARAMETERS 
"""
import torch
import numpy as np

from policies import rollout

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def evaluate(env, policy, agent_name, num_validation_runs=10, episode_length=50):
    success_count = 0
    rewards_suc = 0
    rewards_all = 0
    paths = []
    for k in range(num_validation_runs):
        path = rollout(
                env,
                policy,
                agent_name=agent_name,
                episode_length=episode_length)
        success = sum(path['dones']) > 0
        if success:
            success_count += 1
            rewards_suc += np.sum(path['rewards'])
        rewards_all += np.sum(path['rewards'])
        paths.append(path)
        print(f"test {k}, success {success}, reward {np.sum(path['rewards'])}")
    print("Success rate: ", success_count/num_validation_runs)
    print("Average reward (success only): ", rewards_suc/max(success_count, 1))
    print("Average reward (all): ", rewards_all/num_validation_runs)
    
    return paths, success_count/num_validation_runs, rewards_all/num_validation_runs
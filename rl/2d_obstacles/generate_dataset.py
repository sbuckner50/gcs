# == Imports ==
import numpy as np
import matplotlib.pyplot as plt
import pickle
import scipy.linalg as la
from scipy.spatial import ConvexHull
import seaborn as sns
from pdb import set_trace as debug
import sys

from pydrake.solvers import MosekSolver

from gcs.bezier import BezierGCS
from gcs.linear import LinearGCS
from models.env_2d import vertices
from utils import *

# == Inputs ==
x_start_nom = np.array([.2, .2])
x_goal_nom = np.array([4.8, 4.8])
n_episodes = 100
order = 6
continuity = 2
velocity = np.zeros((2, 2))
regularizer = [1e-1, 1e-1]
hdot_min = 1e-1
qdot_min = -1
qdot_max = 1
dt_sim = 0.02
pos_rad_sample = .2
gcs_augment_degree = 0
base_seed = 5
np.random.seed(base_seed)

# Flags
relaxation = True
savefig = True

# == GCS Setup ==
vertices = augment_gcs_vertices(vertices, degree=gcs_augment_degree)
regions = [make_hpolytope(V) for V in vertices]
region_start = regions[0]
region_goal  = regions[-1]
gcs = BezierGCS(regions, order, continuity, hdot_min=hdot_min)
gcs.setSolver(MosekSolver())
gcs.setPaperSolverOptions()
gcs.addTimeCost(1)
gcs.addVelocityLimits([qdot_min] * 2, [qdot_max] * 2)
if regularizer is not None:
    gcs.addDerivativeRegularization(*regularizer, 2)
x_starts, x_goals = uniform_sample(n_episodes, x_start_nom, x_goal_nom, type='ball', start_rad=pos_rad_sample, goal_rad=pos_rad_sample)

# == Solve SPP for all episodes ==
prev_sample_start = x_start_nom
prev_sample_goal  = x_goal_nom
seeds = np.array(list(range(n_episodes)))
np.random.shuffle(seeds)
times_episodes = []
positions_episodes = []
velocities_episodes = []
accels_episodes = []
max_episode_length = 0
for ep in range(n_episodes):
    gcs.addSourceTarget(x_starts[ep], x_goals[ep], velocity=velocity)
    results = gcs.SolveSingleRandomPath(preprocessing=True, verbose=False, seed=seeds[ep])
    # results = gcs.SolvePath(preprocessing=True, verbose=False)
    traj = results[0]
    n_sim = int(np.floor(traj.end_time() / dt_sim + 1))
    times = np.linspace(traj.start_time(), traj.end_time(), n_sim)
    t_offset = lambda t : max(t+dt_sim/2,0)
    positions = np.squeeze([traj.value(t) for t in times]).T
    velocities = np.squeeze([traj.EvalDerivative(t, derivative_order=1) for t in times]).T
    accels = np.squeeze([traj.EvalDerivative(t_offset(t), derivative_order=2) for t in times]).T
    times_episodes.append(times)
    positions_episodes.append(positions)
    velocities_episodes.append(velocities)
    accels_episodes.append(accels)
    max_episode_length = max(max_episode_length, n_sim)

# == Create dataset ==
dataset = {}
dataset["data"] = []
for ep in range(n_episodes):
    n_sim = times_episodes[ep].size
    episode = {}
    episode["observations"] = np.hstack((positions_episodes[ep].T, velocities_episodes[ep].T))
    episode["actions"] = accels_episodes[ep].T
    episode["rewards"] = np.zeros(n_sim)
    episode["dones"] = np.concatenate((np.zeros(n_sim-1), np.ones(1)))
    dataset["data"].append(episode)

dataset["dt"] = dt_sim
dataset["rad_start_pos"] = pos_rad_sample
dataset["rad_goal_pos"] = pos_rad_sample
dataset["pos_start_nom"] = x_start_nom
dataset["pos_goal_nom"] = x_goal_nom
dataset["max_episode_length"] = max_episode_length
dataset["gcs_augment_degree"] = gcs_augment_degree

filehandler = open("data/2d_obstacles_dataset.pkl", "wb")
pickle.dump(dataset, filehandler)
filehandler.close()

# == Plot control curve comparison for last episode ==
accels_truth = np.squeeze([traj.EvalDerivative(t, derivative_order=2) for t in times]).T
sns.set_theme()
plt.figure(figsize=(3, 3))
plt.plot(times, accels_truth[0,:], linewidth=0.3)
plt.step(times, accels[0,:], linewidth=0.3)
plt.savefig('figures/control_comparison.pdf', bbox_inches='tight')
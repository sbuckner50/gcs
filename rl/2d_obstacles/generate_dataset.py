# == Imports ==
import numpy as np
import matplotlib.pyplot as plt
import pickle
from scipy.spatial import ConvexHull
from pdb import set_trace as debug

from pydrake.solvers import MosekSolver
from pydrake.common._module_py import RandomGenerator

from gcs.bezier import BezierGCS
from gcs.linear import LinearGCS
from models.env_2d import obstacles, vertices

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
dt_sim = .1
base_seed = 5
np.random.seed(base_seed)

# Flags
relaxation = True
savefig = True

# == GCS Setup ==
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

# == Solve SPP for all episodes ==
prev_sample_start = x_start_nom
prev_sample_goal  = x_goal_nom
generator = RandomGenerator(base_seed)
seeds = np.array(list(range(n_episodes)))
np.random.shuffle(seeds)
times_episodes = []
positions_episodes = []
velocities_episodes = []
accels_episodes = []
max_episode_length = 0
for ep in range(n_episodes):
    # Sample boundary conditions
    x_start = region_start.UniformSample(generator, prev_sample_start)
    x_goal = region_goal.UniformSample(generator, prev_sample_goal)
    gcs.addSourceTarget(x_start, x_goal)
    
    # Solve SPP (singular randomized path, no rounding) and store solutions
    results = gcs.SolveSingleRandomPath(preprocessing=True, verbose=False, seed=seeds[ep])
    traj = results[0]
    n_sim = int(np.floor(traj.end_time() / dt_sim + 1))
    times = np.linspace(traj.start_time(), traj.end_time(), n_sim)
    positions = np.squeeze([traj.value(t) for t in times]).T
    velocities = np.squeeze([traj.EvalDerivative(t, derivative_order=1) for t in times]).T
    accels = np.squeeze([traj.EvalDerivative(t, derivative_order=2) for t in times]).T
    times_episodes.append(times)
    positions_episodes.append(positions)
    velocities_episodes.append(velocities)
    accels_episodes.append(accels)
    max_episode_length = max(max_episode_length, n_sim)
    
print("Maximum episode length: " + str(max_episode_length))

# == Create dataset ==
dataset = []
for ep in range(n_episodes):
    episode = {}
    episode["observations"] = np.hstack((positions_episodes[ep].reshape(-1,2), velocities_episodes[ep].reshape(-1,2)))
    episode["actions"] = accels_episodes[ep].reshape(-1,2)
    episode["rewards"] = np.zeros((n_sim,1))
    episode["dones"] = np.vstack((np.zeros((n_sim-1,1)), np.ones((1,1))))
    dataset.append(episode)
    
filehandler = open("data/2d_obstacles_dataset.pkl", "wb")
pickle.dump(dataset, filehandler)
filehandler.close()

# == Plotting ==
x_min = np.min(np.vstack(vertices), axis=0)
x_max = np.max(np.vstack(vertices), axis=0)
setup_fig(x_min, x_max)
plot_environment(obstacles, vertices)
for ep in range(n_episodes):
    alpha = 1/n_episodes
    # plt.scatter(positions_episodes[ep][0][0], positions_episodes[ep][1][0], 500, color='b')
    # plt.scatter(positions_episodes[ep][0][-1], positions_episodes[ep][1][-1], 500, color='b')
    plt.plot(*positions_episodes[ep], 'b', alpha=alpha, linewidth=1, zorder=5)
if savefig:
    plt.savefig('figures/dataset.pdf', bbox_inches='tight')
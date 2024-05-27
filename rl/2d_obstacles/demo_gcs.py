# == Imports ==
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
from pdb import set_trace as debug

from pydrake.geometry.optimization import HPolyhedron
from pydrake.solvers import MosekSolver

from gcs.bezier import BezierGCS
from gcs.linear import LinearGCS
from models.env_2d import obstacles, vertices

from utils import *

# == Inputs ==
x_start = np.array([.2, .2])
x_goal = np.array([4.8, 4.8])
order = 6
continuity = 2
velocity = np.zeros((2, 2))
regularizer = [1e-1, 1e-1]
hdot_min = 1e-1
qdot_min = -1
qdot_max = 1
n_sim = 500
base_seed = 144

# Flags
relaxation = True
savefig = True

# == GCS Setup ==
regions = [make_hpolytope(V) for V in vertices]
gcs = BezierGCS(regions, order, continuity, hdot_min=hdot_min)
gcs.setSolver(MosekSolver())
gcs.setPaperSolverOptions()
gcs.addTimeCost(1)
gcs.addVelocityLimits([qdot_min] * 2, [qdot_max] * 2)
gcs.addSourceTarget(x_start, x_goal, velocity=velocity)
gcs.addDerivativeRegularization(*regularizer, 2)

# == Solve SPP ==
results = gcs.SolvePath(preprocessing=True, verbose=False)
traj = results[0]
times = np.linspace(traj.start_time(), traj.end_time(), n_sim)
positions = np.squeeze([traj.value(t) for t in times]).T

# == Plotting ==
x_min = np.min(np.vstack(vertices), axis=0)
x_max = np.max(np.vstack(vertices), axis=0)
setup_fig(x_min, x_max)
plot_environment(obstacles, vertices)
plt.plot(*positions, 'b', zorder=5)
if savefig:
    plt.savefig('figures/optimal_solution.pdf', bbox_inches='tight')
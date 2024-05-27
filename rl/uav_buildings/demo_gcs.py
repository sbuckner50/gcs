# == Imports ==
# general
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
from pdb import set_trace as debug
import sys

# pydrake
from pydrake.examples import QuadrotorGeometry
from pydrake.geometry import MeshcatVisualizer, Rgba, StartMeshcat
from pydrake.geometry.optimization import HPolyhedron, VPolytope
from pydrake.multibody.plant import AddMultibodyPlantSceneGraph
from pydrake.multibody.parsing import Parser
from pydrake.perception import PointCloud
from pydrake.solvers import GurobiSolver,  MosekSolver
from pydrake.systems.analysis import Simulator
from pydrake.systems.framework import DiagramBuilder

# gcs
from gcs.bezier import BezierGCS
from reproduction.uav.helpers import FlatnessInverter
from reproduction.uav.building_generation import *
from reproduction.util import *

# Local
from utils import *
sys.path.append("../utils/")
from gcs_utils import *
from env import QuadPlannerEnv

# == Inputs ==
# Environment
pos_start_nom = np.array([-1, -1])
pos_goal_nom = np.array([2, 1])
x_start_nom = np.r_[(pos_start_nom-pos_start_nom)*5, 1.]
x_goal_nom = np.r_[(pos_goal_nom-pos_start_nom)*5., 1.]
building_shape = (3, 3)

# Algorithm
order = 6
continuity = 2
velocity = np.zeros((2, 3))
regularization = 1e-3
hdot_min = 1e-1
qdot_min = -1
qdot_max = 1
dt_sim = .02
pos_rad_sample = .2
gcs_augment_degree = 1
base_seed = 144
np.random.seed(base_seed)

# Flags
relaxation = True
savefig = True

# == Visualizer Setup ==
meshcat = StartMeshcat()
meshcat.SetProperty("/Grid", "visible", False)
meshcat.SetProperty("/Axes", "visible", False)
meshcat.SetProperty("/Lights/AmbientLight/<object>", "intensity", 0.8)
meshcat.SetProperty("/Lights/PointLightNegativeX/<object>", "intensity", 0)
meshcat.SetProperty("/Lights/PointLightPositiveX/<object>", "intensity", 0)

# == Building Generation ==
grid, outdoor_edges, wall_edges = generate_grid_world(shape=building_shape, start=pos_start_nom, goal=pos_goal_nom)
regions = compile_sdf(FindModelFile("models/room_gen/building.sdf"), grid, pos_start_nom, pos_goal_nom, outdoor_edges, wall_edges)

# == GCS Setup ==
gcs = BezierGCS(regions, order, continuity, hdot_min=hdot_min)
gcs.setSolver(MosekSolver())
gcs.setPaperSolverOptions()
gcs.addTimeCost(1e-3)
gcs.addPathLengthCost(1)
gcs.addVelocityLimits(-10 * np.ones(3), 10 * np.ones(3))
gcs.addSourceTarget(x_start_nom, x_goal_nom, velocity=velocity, zero_deriv_boundary=3)
gcs.addDerivativeRegularization(regularization, regularization, 2)
gcs.addDerivativeRegularization(regularization, regularization, 3)
gcs.addDerivativeRegularization(regularization, regularization, 4)

# == Solve SPP ==
results = gcs.SolvePath(True, verbose=False, preprocessing=False)
traj = results[0]
n_sim = int(np.floor(traj.end_time() / dt_sim + 1))
times = np.linspace(traj.start_time(), traj.end_time(), n_sim)
positions = np.squeeze([traj.value(t) for t in times]).T
velocities = np.squeeze([traj.EvalDerivative(t, 1) for t in times]).T
accels = np.squeeze([traj.EvalDerivative(t, 2) for t in times]).T

# == Simulate GCS trajectory in environment ==
env = QuadPlannerEnv(0.02, pos_start_nom, pos_goal_nom, 0.2, 0.2, visualizer=False)
env.cur_obs = np.concatenate((x_start_nom, velocity[0,:].reshape(-1)))
states = np.zeros((env.nx, n_sim))
states[:,0] = env.cur_obs
actions = accels
for ii in range(n_sim-1):
    state_next,_,_ = env.step(actions[:,ii].reshape(-1))
    states[:,ii+1] = state_next
success = env.evaluate_done(state_next)

# # == Visualization ==
# view_regions = False
# track_uav = False

# # Build and run Diagram
# builder = DiagramBuilder()
# plant, scene_graph = AddMultibodyPlantSceneGraph(builder, time_step=0.0)

# parser = Parser(plant, scene_graph)
# parser.package_map().Add("gcs", GcsDir())
# model_id = parser.AddModelFromFile(FindModelFile("models/room_gen/building.sdf"))

# plant.Finalize()

# meshcat_cpp = MeshcatVisualizer.AddToBuilder(builder, scene_graph, meshcat)

# animator = meshcat_cpp.StartRecording()
# if not track_uav:
#     animator = None
# traj_system = builder.AddSystem(FlatnessInverter(traj, animator))
# quad = QuadrotorGeometry.AddToBuilder(builder, traj_system.get_output_port(0), scene_graph)

# diagram = builder.Build()
# debug()

# # Set up a simulator to run this diagram
# simulator = Simulator(diagram)
# simulator.set_target_realtime_rate(1.0)

# meshcat.Delete()

# if view_regions:
#     for ii in range(len(regions)):
#         v = VPolytope(regions[ii])
#         meshcat.SetTriangleMesh("iris/region_" + str(ii), v.vertices(),
#                                 ConvexHull(v.vertices().T).simplices.T, Rgba(0.698, 0.67, 1, 0.4))
        
# # Simulate
# end_time = traj.end_time()
# simulator.AdvanceTo(end_time+0.05)
# meshcat_cpp.PublishRecording()
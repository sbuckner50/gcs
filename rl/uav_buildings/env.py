# == Imports ==
# general
import numpy as np
import scipy.linalg as la
import sys
sys.path.append("../utils/")
from gcs_utils import uniform_sample_ball
from pdb import set_trace as debug

# pydrake
from pydrake.examples import QuadrotorGeometry
from pydrake.geometry import MeshcatVisualizer, Rgba, StartMeshcat
from pydrake.multibody.plant import AddMultibodyPlantSceneGraph
from pydrake.multibody.parsing import Parser
from pydrake.perception import PointCloud
from pydrake.solvers import GurobiSolver,  MosekSolver
from pydrake.systems.analysis import Simulator
from pydrake.systems.framework import DiagramBuilder
from pydrake.systems.framework import LeafSystem
from pydrake.systems.primitives import LogVectorOutput
from pydrake.math import RigidTransform, RollPitchYaw

# gcs
from reproduction.util import *

# class QuadPlannerSystem(LeafSystem):
#     def __init__(self, animator, t_offset=0):
#         LeafSystem.__init__(self)
#         self.obs_port = self.DeclareVectorInputPort("observation", 6)
#         self.act_port = self.DeclareVectorInputPort("action", 3)
#         self.state_port = self.DeclareVectorOutputPort("state", 12, self.CalcState, {self.time_ticket()})
#         self.t_offset = t_offset
#         self.animator = animator

#     def CalcState(self, context, output):
#         obs = self.obs_port.Eval(context)
#         act = self.act_port.Eval(context)

#         q = obs[:3]
#         q_dot = obs[3:]
#         q_ddot = act

#         fz = np.sqrt(q_ddot[0]**2 + q_ddot[1]**2 + (q_ddot[2] + 9.81)**2)
#         r = np.arcsin(-q_ddot[1]/fz)
#         p = np.arcsin(q_ddot[0]/fz)
#         output.set_value(np.concatenate((q, [r, p, 0], q_dot, np.zeros(3))))

#         if self.animator is not None:
#             frame = self.animator.frame(context.get_time())
#             self.animator.SetProperty(frame, "/Cameras/default/rotated/<object>", "position", [-2.5, 4, 2.5])
#             self.animator.SetTransform(frame, "/drake", RigidTransform(-q))

class QuadPlannerSystem(LeafSystem):
    def __init__(self, animator, t_offset=0, dt=0.02):
        LeafSystem.__init__(self)
        state_index = self.DeclareDiscreteState(12)
        self.DeclareStateOutputPort("state", state_index)
        self.DeclarePeriodicDiscreteUpdateEvent(
            period_sec = dt,
            offset_sec = t_offset,
            update=self.CalcState
        )
        self.action_port = self.DeclareVectorInputPort("action", 3)
        self.animator = animator

    def CalcState(self, context, output):

        action = self.act_port.Eval(context)
        state = context.get_discrete_state_vector().GetAtIndex(0)
        
        q = state[0:3]
        q_dot = state[6:9]
        q_ddot = action

        fz = np.sqrt(q_ddot[0]**2 + q_ddot[1]**2 + (q_ddot[2] + 9.81)**2)
        r = np.arcsin(-q_ddot[1]/fz)
        p = np.arcsin(q_ddot[0]/fz)
        output.set_value(np.concatenate((q, [r, p, 0], q_dot, np.zeros(3))))

        if self.animator is not None:
            frame = self.animator.frame(context.get_time())
            self.animator.SetProperty(frame, "/Cameras/default/rotated/<object>", "position", [-2.5, 4, 2.5])
            self.animator.SetTransform(frame, "/drake", RigidTransform(-q))

class QuadPlannerEnv():
    def __init__(self, dt, pos_start_nom, pos_goal_nom, rad_start_pos, rad_goal_pos, visualizer=True) -> None:
        # Internal variables
        self.dt = dt
        self.nx = 6
        self.nu = 3
        self.cur_time = 0
        self.cur_obs = np.zeros(self.nx)
        self.pos_start_nom = pos_start_nom
        self.pos_goal_nom = pos_goal_nom
        self.rad_start_pos = rad_start_pos
        self.rad_goal_pos = rad_goal_pos

        # Drake setup: visualizer
        if visualizer:
            self.meshcat = StartMeshcat()
            self.meshcat.SetProperty("/Grid", "visible", False)
            self.meshcat.SetProperty("/Axes", "visible", False)
            self.meshcat.SetProperty("/Lights/AmbientLight/<object>", "intensity", 0.8)
            self.meshcat.SetProperty("/Lights/PointLightNegativeX/<object>", "intensity", 0)
            self.meshcat.SetProperty("/Lights/PointLightPositiveX/<object>", "intensity", 0)

        # Drake setup: diagram
        self.builder = DiagramBuilder()
        self.plant, self.scene_graph = AddMultibodyPlantSceneGraph(self.builder, time_step=self.dt)
        self.parser = Parser(self.plant, self.scene_graph)
        self.parser.package_map().Add("gcs", GcsDir())
        self.model_id = self.parser.AddModelFromFile(FindModelFile("models/room_gen/building.sdf"))
        self.plant.Finalize()
        self.system = self.builder.AddSystem(QuadPlannerSystem(None))
        self.quad_geometry = QuadrotorGeometry.AddToBuilder(self.builder, self.system.get_output_port(0), self.scene_graph)
        # self.logger = LogVectorOutput(self.system.get_output_port(0), self.builder)
        self.context = self.system.CreateDefaultContext()
        self.diagram = self.builder.Build()

        # Drake setup: simulator
        self.simulator = Simulator(self.diagram)
        if visualizer:
            self.meshcat.Delete()

    def reset(self):
        """
        Sample initial condition and set that as the current observation
        """
        init_pos = uniform_sample_ball(self.pos_start_nom, self.rad_start_pos)
        init_vel = np.zeros(2)
        self.cur_obs = np.concatenate((init_pos,init_vel))
        return self.cur_obs
        
    def step(self, action):
        """
        Step through the simulator
        """
        self.context.SetTime(self.cur_time)
        self.system.GetInputPort("observation").FixValue(self.context, self.cur_obs)
        self.system.GetInputPort("action").FixValue(self.context, action)
        self.simulator.AdvanceTo(self.cur_time + self.dt)
        debug()
        new_state = self.system.GetOutputPort("state").Eval(self.context)
        new_obs = np.concatenate((new_state[0:3], new_state[6:9]))
        reward = 0 # TODO: make this something tangible for model-free RL
        done = self.evaluate_done(new_obs)
        self.cur_obs = new_obs
        self.cur_time += self.dt

        return new_obs, reward, done
        
    def render(self):
        return 0 # TODO: make this some sort of animation if desired
    
    def evaluate_done(self, obs, buffer=0.2):
        """
        Check if observation falls within the terminal condition boundaries
        """
        pos = obs[:2,]
        done = la.norm(pos - self.pos_goal_nom) <= (1+buffer)*self.rad_goal_pos
        return done
        
    def evaluate_runaway_error(self, obs):
        """
        Checks if the observation has become practically infeasible
        """
        obs_runaway = [abs(obs[c,0]) >= 100 for c in range(self.nx)]
        error = any(obs_runaway)
        return error
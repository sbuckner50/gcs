import numpy as np
import scipy.linalg as la
import sys
sys.path.append("../utils/")
from gcs_utils import uniform_sample_ball
from pdb import set_trace as debug

class Obstacle2DEnv():
    def __init__(self, dt, pos_start_nom, pos_goal_nom, rad_start_pos, rad_goal_pos) -> None:
        self.dt = dt
        self.A = np.block([
            [np.zeros((2,2)), np.eye(2)],
            [np.zeros((2,2)), np.zeros((2,2))]
        ])
        self.B = np.block([
            [np.zeros((2,2))],
            [np.eye(2)]
        ])
        self.nx = 4
        self.nu = 2
        self.cur_time = 0
        self.cur_obs = np.zeros((self.nx,1))
        self.dyn = lambda t,x,u : self.A @ x + self.B @ u # LTI double integrator dynamics
        self.pos_start_nom = pos_start_nom
        self.pos_goal_nom = pos_goal_nom
        self.rad_start_pos = rad_start_pos
        self.rad_goal_pos = rad_goal_pos
        
    def reset(self):
        """
        Sample initial condition and set that as the current observation
        """
        init_pos = uniform_sample_ball(self.pos_start_nom, self.rad_start_pos)
        init_vel = np.zeros(2)
        self.cur_obs = np.concatenate((init_pos,init_vel)).reshape(-1,1)
        return self.cur_obs
        
    def step(self, action):
        """
        Perform Runge-Kutta 4th Order (RK4) integration on the dynamics.
        """
        tc = self.cur_time
        xc = self.cur_obs
        uc = action
        h = self.dt
        
        k1 = self.dyn(tc,       xc,            uc)
        k2 = self.dyn(tc + h/2, xc + h * k1/2, uc)
        k3 = self.dyn(tc + h/2, xc + h * k2/2, uc)
        k4 = self.dyn(tc + h,   xc + h * k3,   uc)
        
        xn = (xc + h/6 * (k1 + 2*k2 + 2*k3 + k4))
        new_obs = xn
        reward = 0 # TODO: make this non-trivial once we decide a good reward fn
        done = self.evaluate_done(xn)
        self.cur_time += h
        self.cur_obs = new_obs
        
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
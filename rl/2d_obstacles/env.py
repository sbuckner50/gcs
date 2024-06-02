import numpy as np
import scipy.linalg as la
import sys
from utils import *
from models.env_2d import obstacles, obstacles_poly, vertices
from pdb import set_trace as debug

class Obstacle2DEnv():
    def __init__(self, dt, pos_start_nom, pos_goal_nom, rad_start_pos, rad_goal_pos, gcs_augment_degree) -> None:
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
        
        # reward hyperpameters
        self.w_distance = 3
        self.w_safe = 1
        
        # Generate (buffered) obstacles
        self.obstacles_buffered_vertices = obstacles
        self.obstacles_buffered = [make_hpolytope(V) for V in self.obstacles_buffered_vertices]
        
        # Generate true obstacles
        # TODO: for now, just treat these as the buffered obstacles (equal to not buffering)
        self.obstacles_vertices = self.obstacles_buffered_vertices
        self.obstacles = self.obstacles_buffered
        # self.obstacles = [O.Scale(1-obstacle_buffer) for O in self.obstacles_buffered]
        # self.obstacles_vertices = [get_hpolytope_vertices(O) for O in self.obstacles]
        
        # Generate GCS segments based on desired augment degree
        self.regions_vertices = augment_gcs_vertices(vertices, degree=gcs_augment_degree)
        self.regions = [make_hpolytope(V) for V in self.regions_vertices]

        # Generate obstacle matrices (on convex polytopic obstacles only)
        self.obstacles_poly = [make_hpolytope(V) for V in obstacles_poly]
        self.n_obstacles = len(self.obstacles_poly)
        self.A_obstacles = [O.A() for O in self.obstacles_poly]
        self.b_obstacles = [O.b() for O in self.obstacles_poly]
        
        # Compute convex hull of obstacles + GCS sets
        # Use this to "block in" the observation space
        stacked_set_vertices = self.regions_vertices[0]
        for k in range(1,len(self.regions_vertices)):
            stacked_set_vertices = np.vstack((stacked_set_vertices, self.regions_vertices[k]))
        for k in range(0,len(self.obstacles_vertices)):
            stacked_set_vertices = np.vstack((stacked_set_vertices, self.obstacles_vertices[k]))
        stacked_set_polytope = make_hpolytope(stacked_set_vertices)
        
        # Make each side registered as a separate obstacle
        A_closure = -stacked_set_polytope.A() # negative so that anything inside is considered safe (opposite of a traditional obstacle)
        b_closure = -stacked_set_polytope.b()
        n_closures = A_closure.shape[0]
        self.n_obstacles += n_closures
        self.A_obstacles += [A_closure[k,:] for k in range(n_closures)]
        self.b_obstacles += [b_closure[k] for k in range(n_closures)]

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
        reward = self.evaluate_reward(new_obs, action)
        done = self.evaluate_done(new_obs)
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
        
    def evaluate_collision(self, obs, agent_name='bc'):
        """
        Checks if the observation has resulted in a collision
        (or has significantly exceeded the observation space)
        """
        obs_runaway = [abs(obs[c,0]) >= 100 for c in range(self.nx)]
        error = any(obs_runaway)
        if agent_name != 'bc':
            safe,_ = self.evaluate_obstacle_safety(obs)
            error = error or not safe
        return error
    
    def evaluate_obstacle_safety(self, obs):
        # Determine if observation falls outside of all obstacles, and the closest margin
        pos = obs[:2].reshape(-1,1)
        safe = True
        min_margin = 1e6 # arbitrarily large number to start
        for k in range(self.n_obstacles):
            A = self.A_obstacles[k]
            b = self.b_obstacles[k].reshape(-1,1)
            LHS = A@pos - b

            # Check if within obstacle
            if all(LHS < 0):
                safe = False
                min_margin = 0
                break
            else:
                margin = la.norm(LHS[LHS >= 0])
                min_margin = min(min_margin, margin)
                
        return safe, min_margin
    
    def evaluate_reward(self, obs, act):
        pos = obs[:2,]
        w_sum = self.w_distance + self.w_safe
        w_distance = self.w_distance / w_sum
        w_safe = self.w_safe / w_sum
        inv_distance_goal = min(1., self.rad_goal_pos/la.norm(pos - self.pos_goal_nom))
        # inv_distance_goal = - la.norm(pos - self.pos_goal_nom)
        _,safety_margin = self.evaluate_obstacle_safety(obs)
        reward = w_distance * inv_distance_goal + w_safe * safety_margin
        return reward
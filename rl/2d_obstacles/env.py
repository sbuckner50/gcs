import numpy as np
from pdb import set_trace as debug

class Obstacle2DEnv():
    def __init__(self, init_conds, term_conds, dt) -> None:
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
        self.init_min = np.min(init_conds, axis=0)
        self.init_max = np.max(init_conds, axis=0)
        self.term_min = np.min(term_conds, axis=0)
        self.term_max = np.max(term_conds, axis=0)
        
    def reset(self):
        """
        Sample initial condition and set that as the current observation
        """
        init_obs = np.array([np.random.uniform(self.init_min[c], self.init_max[c]) for c in range(self.nx)]).reshape(-1,1)
        self.cur_obs = init_obs
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
    
    def evaluate_done(self, obs):
        """
        Check if observation falls within the terminal condition boundaries
        """
        obs_contained = [obs[c,0] >= self.term_min[c] and obs[c,0] <= self.term_max[c] for c in range(self.nx)]
        done = all(obs_contained)
        return done
        
    def evaluate_runaway_error(self, obs):
        """
        Checks if the observation has become practically infeasible
        """
        obs_runaway = [abs(obs[c,0]) >= 100 for c in range(self.nx)]
        error = any(obs_runaway)
        return error
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
import contextlib
from pydrake.geometry.optimization import HPolyhedron
from pydrake.common._module_py import RandomGenerator
from pdb import set_trace as debug

def make_hpolytope(V):
    ch = ConvexHull(V)
    return HPolyhedron(ch.equations[:, :-1], - ch.equations[:, -1])

def uniform_sample(
    count, 
    x_start_nom, 
    x_goal_nom, 
    type='ball', 
    start_rad=0.2, 
    goal_rad=0.2,
    start_gcs_set=None,
    goal_gcs_set=None):
    x_starts = []
    x_goals = []
    prev_sample_start = x_start_nom
    prev_sample_goal = x_goal_nom
    for _ in range(count):
        if type=='ball':
            x_starts.append(uniform_sample_ball(x_start_nom, start_rad))
            x_goals.append(uniform_sample_ball(x_goal_nom, goal_rad))
        elif type=='graph':
            x_starts.append(start_gcs_set.UniformSample(generator, prev_sample_start))
            x_goals.append(goal_gcs_set.UniformSample(generator, prev_sample_goal))
    return x_starts, x_goals

def uniform_sample_ball(x0, rad):
    ang = np.random.uniform(0, 2*3.14159)
    mag = np.random.uniform(0, rad)
    return np.array([
        x0[0] + mag*np.sin(ang),
        x0[1] + mag*np.cos(ang)
    ])

def augment_gcs_vertices(vertices, degree=1):
    if degree == 0:
        return vertices
    elif degree == 1:
        vertices_new = []
        vertices_new.append(vertices[0])
        for V in vertices[1:-1]:
            polytope = make_hpolytope(V)
            center = polytope.ChebyshevCenter()
            for k in range(V.shape[0]):
                kp1 = (k + 1) % V.shape[0]
                V_new = np.vstack((V[k,:], V[kp1,:], center))
                vertices_new.append(V_new)
        vertices_new.append(vertices[-1])
        return vertices_new
    else:
        ValueError("Degree must be either 0 or 1 for now.")
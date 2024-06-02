import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
import contextlib
from pydrake.geometry.optimization import HPolyhedron, VPolytope
from pydrake.common._module_py import RandomGenerator
from pdb import set_trace as debug
from models.env_2d import vertices

generator = RandomGenerator(0)
plt.rcParams['text.usetex'] = True

@contextlib.contextmanager
def temp_seed(seed):
    state = np.random.get_state()
    np.random.seed(seed)
    try:
        yield
    finally:
        np.random.set_state(state)

def setup_fig(ax=None, vertices=vertices):
    x_min = np.min(np.vstack(vertices), axis=0)
    x_max = np.max(np.vstack(vertices), axis=0)
        
    if ax == None:
        plt.figure(figsize=(3, 3))
        ax = plt.gca()

    ax.set_xlim([x_min[0], x_max[0]])
    ax.set_ylim([x_min[1], x_max[1]])
    
    tick_gap = .2
    n_ticks = lambda x_min, x_max: round((x_max - x_min) / tick_gap) + 1
    x_ticks = np.linspace(x_min[0], x_max[0], n_ticks(x_min[0], x_max[0]))
    y_ticks = np.linspace(x_min[1], x_max[1], n_ticks(x_min[1], x_max[1]))
    ax.set_xticks(x_ticks)
    ax.set_yticks(y_ticks)
    
    label_gap = .5
    keep_label = lambda t: np.isclose(t % label_gap, 0) or np.isclose(t % label_gap, label_gap)
    x_labels = [int(t) if keep_label(t) else '' for t in x_ticks]
    y_labels = [int(t) if keep_label(t) else '' for t in y_ticks]
    ax.set_xticklabels(x_labels)
    ax.set_yticklabels(y_labels)
    ax.set_xlabel(r"$x$-position")
    ax.set_ylabel(r"$y$-position")
    # ax.set_xlabel("x")
    # ax.set_ylabel("y")
    
    plt.axis('square')
    # plt.grid()

def plot_environment(env, ax=plt):
    # Plot obstacles
    for O in env.obstacles_vertices:
        ax.fill(*O.T, fc=[1,.7,.7], ec=[1,.3,.3], zorder=4, linestyle='-', linewidth=1)

    # Plot regions
    scales = np.linspace(.8,1,len(env.regions_vertices))
    with temp_seed(0):
        np.random.shuffle(scales)
    for k,V in enumerate(env.regions_vertices):
        scale = scales[k]
        color = [scale, scale, scale]
        color2 = [.5,.5,.5]
        # color2 = None
        ax.fill(*V.T, fc=color, ec=color2, zorder=3)

def make_hpolytope(V):
    ch = ConvexHull(V)
    return HPolyhedron(ch.equations[:, :-1], - ch.equations[:, -1])

def get_hpolytope_vertices(H):
    V = VPolytope(H).vertices().T

    # TODO: hacky hardcoded reshuffling for plt.fill to work properly, make sure to update this for general use
    V_shuffled = np.copy(V)
    if V.shape[0] == 4:
        V_shuffled[1,:] = V[0,:]
        V_shuffled[0,:] = V[1,:]
    elif V.shape[0] > 4:
        debug()
    
    return V_shuffled

def uniform_sample(
    count, 
    x_start_nom, 
    x_goal_nom, 
    type='ball', 
    start_rad=0.2, 
    goal_rad=0.2,
    start_gcs_set=None,
    goal_gcs_set=None,
    dim=2):
    x_starts = []
    x_goals = []
    prev_sample_start = x_start_nom
    prev_sample_goal = x_goal_nom
    for _ in range(count):
        if type=='ball':
            x_starts.append(uniform_sample_ball(x_start_nom, start_rad, dim=dim))
            x_goals.append(uniform_sample_ball(x_goal_nom, goal_rad, dim=dim))
        elif type=='graph':
            x_starts.append(start_gcs_set.UniformSample(generator, prev_sample_start))
            x_goals.append(goal_gcs_set.UniformSample(generator, prev_sample_goal))
    return x_starts, x_goals

def uniform_sample_ball(x0, rad, dim=2):
    ang = np.random.uniform(0, 2*3.14159)
    mag = np.random.uniform(0, rad)
    ball = np.array([x0[0] + mag*np.sin(ang),
                     x0[1] + mag*np.cos(ang)])
    if dim == 3:
        ball = np.concatenate((ball, np.array([0])))
    return ball

def augment_gcs_vertices(vertices, degree=1):
    if degree == 0:
        return vertices
    elif degree == 1:
        vertices_new = []
        vertices_new.append(vertices[0])
        for V in vertices[1:-1]:
            polytope = make_hpolytope(V)
            center = polytope.MaximumVolumeInscribedEllipsoid().center()
            for k in range(V.shape[0]):
                kp1 = (k + 1) % V.shape[0]
                V_new = np.vstack((V[k,:], V[kp1,:], center))
                vertices_new.append(V_new)
        vertices_new.append(vertices[-1])
        return vertices_new
    else:
        ValueError("Degree must be either 0 or 1 for now.")
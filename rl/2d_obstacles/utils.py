import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
import contextlib
from pydrake.geometry.optimization import HPolyhedron

@contextlib.contextmanager
def temp_seed(seed):
    state = np.random.get_state()
    np.random.seed(seed)
    try:
        yield
    finally:
        np.random.set_state(state)

def make_hpolytope(V):
    ch = ConvexHull(V)
    return HPolyhedron(ch.equations[:, :-1], - ch.equations[:, -1])

def setup_fig(x_min, x_max):
    plt.figure(figsize=(3, 3))
    plt.axis('square')

    plt.xlim([x_min[0], x_max[0]])
    plt.ylim([x_min[1], x_max[1]])
    
    tick_gap = .2
    n_ticks = lambda x_min, x_max: round((x_max - x_min) / tick_gap) + 1
    x_ticks = np.linspace(x_min[0], x_max[0], n_ticks(x_min[0], x_max[0]))
    y_ticks = np.linspace(x_min[1], x_max[1], n_ticks(x_min[1], x_max[1]))
    plt.xticks(x_ticks)
    plt.yticks(y_ticks)
    
    label_gap = .5
    keep_label = lambda t: np.isclose(t % label_gap, 0) or np.isclose(t % label_gap, label_gap)
    x_labels = [int(t) if keep_label(t) else '' for t in x_ticks]
    y_labels = [int(t) if keep_label(t) else '' for t in y_ticks]
    plt.gca().set_xticklabels(x_labels)
    plt.gca().set_yticklabels(y_labels)
    # plt.xlabel("$x$", "interpreter", "latex")
    # plt.ylabel("$y$", "interpreter", "latex")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.grid()

def plot_environment(obstacles, vertices):
    for O in obstacles:
        plt.fill(*O.T, fc='lightcoral', ec='k', zorder=4)
        
    scales = np.linspace(.7,1,len(vertices))
    with temp_seed(0):
        np.random.shuffle(scales)
    for k,V in enumerate(vertices):
        scale = scales[k]
        color = [scale, scale, scale]
        color2 = [.5,.5,.5]
        plt.fill(*V.T, fc=color, ec=color2, zorder=3)
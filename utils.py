from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from matplotlib import pyplot as plt


def visualize_3D(x, y, z):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(xs, ys, zs)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    return ax

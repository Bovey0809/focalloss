import torch.nn.functional as F
import torch
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import random
from mpl_toolkits.mplot3d import Axes3D
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
x = np.arange(-4, 4, 0.1)
_y, _x = np.meshgrid(x, x)

grid = np.stack([_x, _y], 0)
grid = grid.reshape(2, -1).transpose(1, 0)




# visualize
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

xs = grid[:, 0]
ys = grid[:, 1]
zs = gau(grid)
zs = np.where(zs > 0.1, zs, np.zeros_like(zs))
ax.scatter(xs, ys, zs)

ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
plt.show()


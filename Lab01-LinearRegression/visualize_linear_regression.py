import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

def file_read(path):
    with open(path, 'r') as label:
        X = []
        Y = []

        for line in label:
            line = line.split()
            X.append(float(line[0]))
            Y.append(float(line[1]))

        X = np.asarray(X).astype(np.float32)
        Y = np.asarray(Y).astype(np.float32)

        return X, Y

M = np.arange(0, 0.6, 0.01)
C = np.arange(30., 90., 1)

X, Y = file_read("train.txt")

M_grid, C_grid = np.meshgrid(M, C)
Cost = np.zeros_like(M_grid)
M_grid = np.zeros_like(M_grid)
C_grid = np.zeros_like(C_grid)


for i_m, m in enumerate(M):
    for i_c, c in enumerate(C):
        M_grid[i_m][i_c] = m
        C_grid[i_m][i_c] = c
        Cost[i_m][i_c] = np.mean((Y-(m*X+c))**2)

i_m, i_c = np.unravel_index(Cost.argmin(), Cost.shape)

fig = plt.figure()
ax = plt.axes(projection='3d')
ax.plot_surface(M_grid, C_grid, Cost, cmap='jet')
ax.set_xlabel('m')
ax.set_ylabel('c')
ax.set_zlabel('cost')

plt.show()
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt


M = np.arange(-1., 1., 0.01)
C = np.arange(-1., 1., 0.01)

X = [1., 2., 3., 9., 10., 15.]
Y = [0., 0., 0., 1., 1., 1.]

X = np.array(X)
Y = np.array(Y)

M_grid, C_grid = np.meshgrid(M, C)
MSE_Cost = np.zeros_like(M_grid)
BCE_Cost = np.zeros_like(M_grid)
M_grid = np.zeros_like(M_grid)
C_grid = np.zeros_like(C_grid)

for i_m, m in enumerate(M):
    for i_c, c in enumerate(C):
        M_grid[i_m][i_c] = m
        C_grid[i_m][i_c] = c
        Y_pred = 1./(1+np.exp(-(m*X+c))) # sigmoid function
        MSE_Cost[i_m][i_c] = np.mean((Y-(Y_pred))**2)
        BCE_Cost[i_m][i_c] = -np.mean( Y* np.log(Y_pred) + (1-Y)*np.log(1-Y_pred))


plt.figure(1)
ax = plt.axes(projection='3d')
ax.plot_surface(M_grid, C_grid, BCE_Cost, cmap='jet')
ax.set_xlabel('m')
ax.set_ylabel('c')
ax.set_zlabel('BCE_cost')

plt.figure(2)
ax = plt.axes(projection='3d')
ax.plot_surface(M_grid, C_grid, MSE_Cost, cmap='jet')
ax.set_xlabel('m')
ax.set_ylabel('c')
ax.set_zlabel('mse_cost')

plt.figure(3)
plt.scatter(X, Y)
plt.show()
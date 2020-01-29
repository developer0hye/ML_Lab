import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def file_read(path):
    with open(path, 'r') as label:
        X = []

        Y = []

        for line in label:
            line = line.split()

            X.append(float(line[0]))
            X.append(float(line[1]))

            Y.append(float(line[2]))

        X = np.asarray(X).astype(np.float32)
        X = X.reshape(-1, 2)

        Y = np.asarray(Y).astype(np.float32)

        return X, Y

X, Y = file_read("train.txt")

fig = plt.figure()
ax = plt.axes(projection='3d')

ax.scatter(X[..., 0], X[..., 1], Y, c = Y, s= 50, alpha=0.5)

ax.set_xlabel('mid term point')
ax.set_ylabel('final term point')
ax.set_zlabel('total point')
plt.show()

import numpy as np
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

def scatter(x, y, c):
    for xi, yi in zip(x,y):
        plt.scatter(xi, yi, color=c)

X, Y = file_read("train.txt")

b_students = Y[...]==1
a_students = Y[...]==0

scatter(X[a_students], Y[a_students], 'b')
scatter(X[b_students], Y[b_students], 'r')

plt.xlabel("Point")
plt.ylabel("Grade(A:1, B:0)")
plt.show()
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



X, Y = file_read("train.txt")

a_students = Y[...]==1
b_students = Y[...]==0

plt.scatter(X[a_students], Y[a_students], color='r', label='A')
plt.scatter(X[b_students], Y[b_students], color='b', label='B')

plt.xlabel("Point")
plt.ylabel("Grade(A:1, B:0)")
plt.legend()

plt.show()

import sys
import scipy.io as sio
from pprint import pprint
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

mat_fname = "data_logistic.mat"
mat_dict = sio.loadmat(mat_fname)

samples = mat_dict["logistic_data"]


# m = features number
# n = samples number
# hθ = sigmond(θT x)


# ‫‬J(θ) = -1/m * sum(Y(i) * log(hθ(X(i)) - (1 - Y(i)) * log(1 - hθ(X(i)))))
# θj := θj - alpha * dJ/dθj = θj - alpha/m * (hθ(X(i) - Y(i)) * X(i)j
# j = 0,1,..,n 

def sigmoid(z):
	return 1 / (1 + np.exp(-z))

def loss(h, y):
    return (-y * np.log(h) - (1 - y) * np.log(1 - h)).mean()


X = []
Y = []
m = 0
for sample in samples:
	X.append(sample[0:2])
	Y.append(sample[2])
	m += 1
X = np.array(X)
Y = np.array(Y)


alpha = 0.01
num_iter = 10000
theta = np.zeros(X.shape[1])

for i in range(num_iter):
	z = np.dot(X, theta)
	h = sigmoid(z)
	gradient = np.dot(np.array(X).T, (h - Y)) / m
	theta -= alpha * gradient
    
z = np.dot(X, theta)
h = sigmoid(z)
print('Optimal Loss: ', loss(h, Y), '\n')
print('Theta: ', theta)




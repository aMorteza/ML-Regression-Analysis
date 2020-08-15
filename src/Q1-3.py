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

# ‫‬J(θ) = -1/m * sum(Y(i) * log(hθ(X(i)) - (1 - Y(i)) * log(1 - hθ(X(i))))) + L * sum(θj ^ 2)
# θj :=  θj * (1 - (alpha * L) /m) - alpha/m * (hθ(X(i) - Y(i)) * X(i)j
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
L = 0.01 #L2-norm effect here

for i in range(num_iter):
	z = np.dot(X, theta)
	h = sigmoid(z)
	gradient = np.dot(X.T, (h - Y)) / m
	theta = theta * (1 - (alpha * L) / m) - alpha * gradient  #L2-norm effect here
    
z = np.dot(X, theta)
h = sigmoid(z)

term = 0
for t in theta:
	term = t**2
term = L * term #L2-norm effect here

print(' L:', L, '\n\n Optimal Loss:', loss(h, Y) + term, '\n')
print(' Theta: ', theta)




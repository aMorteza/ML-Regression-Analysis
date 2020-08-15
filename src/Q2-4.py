import sys
import scipy.io as sio
from pprint import pprint
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

# h(X(i)) = theta_0 + theta_1 * X1(i) + theta_2 * X2(i) 

def normalize(x):
	row_sums = x.sum(axis=1)
	return x / row_sums[:, np.newaxis]
   	

def hypothesis(theta, X, n):
	m = X.shape[0]
	h = np.ones((m, 1))
	theta = theta.reshape(1, n+1)
	for i in range(0, m):
		h[i] = float(np.matmul(theta, X[i]))
	return h.reshape(m)
    

def BatchGD(theta, alpha, num_iters, h, X, y, n):
    m = X.shape[0]
    cost = np.ones(num_iters)
    for i in range(0, num_iters):
        theta[0] = theta[0] - (alpha/m) * sum(h - y)
        for j in range(1, n+1):
            theta[j] = theta[j] - (alpha/m) * sum((h - y) * X.transpose()[j])
        h = hypothesis(theta, X, n)
        cost[i] = (1/m) * 0.5 * sum(np.square(h - y))
    theta = theta.reshape(1, n+1)
    return theta, cost  

x = np.array([[0, 1], [5, 1], [15, 2], [25, 5], [35, 11], [45, 15], [55, 34], [60, 35]]) 
X = normalize(x)
y = np.array([4, 5, 20, 14, 32, 22, 38, 43]) 

n = X.shape[1]
m = X.shape[0]
alpha = 0.03
num_iters = 20000

one_column = np.ones((m,1))
X = np.concatenate((one_column, X), axis = 1)
# initializing vector
theta = np.zeros(n+1)
# hypothesis calculation
h = hypothesis(theta, X, n)
# optimizing theta by Gradient Descent
theta, cost = BatchGD(theta, alpha, num_iters, h, X, y, n)


theta_0 = theta[0][0]
theta_1 = theta[0][1]
theta_2 = theta[0][2]

print("theta0:", theta_0, "\ntheta1:", theta_1, "\ntheta2:", theta_2)

error = y - (theta[0][0] + theta[0][1] * X[:, 0] + theta[0][2] * X[:, 1])
print("Error:", error)

y_pred = np.ones((m, 1))
theta = theta.reshape(1, n+1)
for i in range(0, m):
	y_pred[i] = float(np.matmul(theta, X[i]))
y_pred = y_pred.reshape(m)

fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(x[:, 0], x[:, 1], y, zdir='z', color= 'r')
ax.plot(x[:, 0], x[:, 1], y_pred, zdir='z', color= 'b')
ax.set_title("Multiple linear regression", fontsize=16)
ax.set_xlabel("$X_1$", fontsize=14)
ax.set_ylabel("$X_2$", fontsize=14)
ax.set_zlabel('Y', fontsize=14)

plt.legend(["h(x)", "(X, y)"], loc='best')   
plt.show()




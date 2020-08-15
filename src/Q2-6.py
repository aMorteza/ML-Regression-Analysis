import sys
import scipy.io as sio
from pprint import pprint
import matplotlib.pyplot as plt
import numpy as np

# y = a + b1x + b2^2 + e

def normalize(x):
   return (x - x.min())/(x.max() - x.min())

x = np.array([5, 15, 25, 35, 45, 55]) 
X = normalize(x)
y = np.array([15, 11, 2, 8, 25, 32])

theta = np.poly1d(np.polyfit(X, y, 2))
myline = np.linspace(X.min(), X.max(), y.max())

print(theta)
print("theta_0:", theta[0], "\ntheta_1:", theta[1], "\ntheta_2:",theta[2])

y_pred = theta[0] + theta[1] * X + theta[2] * X**2

print("Error:", y - y_pred)

plt.figure(figsize=(8, 6))
plt.scatter(X, y, color = "r", marker = "o", s = 30) 
plt.title("Predicted Regression Line", fontsize=16)
plt.plot(X, y_pred, color = "b") 
plt.legend(['h(x)', '(x, y)'], loc='best')

plt.xlabel('X', fontsize=14) 
plt.ylabel('Y', fontsize=14)   
plt.show()

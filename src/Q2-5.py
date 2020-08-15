import sys
import scipy.io as sio
from pprint import pprint
import matplotlib.pyplot as plt
import numpy as np

# h(X(i)) = theta_0 + theta_1 * X1(i)

def normalize(x):
   return (x - x.min())/(x.max() - x.min())
    
# theta = (X.T X)^(-1)(X.T)(Y)    
def normal_equation(X, y):
    theta = []
    m = np.size(x) 
    bias = np.ones((m, 1))
    X = np.reshape(X, (m, 1))
    X = np.append(bias, X, axis=1)
    theta = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)
    return theta

x = np.array([5, 15, 25, 35, 45, 55]) 
X = normalize(x)
y = np.array([15, 11, 2, 8, 25, 32])

theta = normal_equation(X, y)
print("theta_0", theta[0], "\ntheta_1", theta[1])

y_pred = theta[0] + theta[1] * X 
error = y - (y_pred)
print("Error:", error)

plt.figure(figsize=(8, 6))
plt.scatter(X, y, color = "r", marker = "o", s = 30) 
plt.title("Predicted Regression Line", fontsize=16)
plt.plot(X, y_pred, color = "b") 
plt.legend(['h(x)', '(x, y)'], loc='best')

plt.xlabel('X', fontsize=14) 
plt.ylabel('Y', fontsize=14)   
plt.show() 






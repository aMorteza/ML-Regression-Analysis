import sys
import scipy.io as sio
from pprint import pprint
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

def estimate_coef(x, y): 
    n = np.size(x) 
    m_x, m_y = np.mean(x), np.mean(y) 
    # calculating regression coefficients
    Sxy = np.sum(y*x) - n*m_y*m_x
    Sxx = np.sum(x*x) - n*m_x*m_x
    theta_1 =  Sxy/Sxx  
    theta_0 = m_y - theta_1*m_x 
    return(theta_0, theta_1) 
  

x = np.array([5, 15, 25, 35, 45, 55]) 
y = np.array([5, 20, 14, 32, 22, 38]) 

x = (x - x.min())/(x.max() - x.min())
   
theta = estimate_coef(x, y) 
print("theta0:", theta[0], " theta1:", theta[1])

error = y - (theta[0] + theta[1] * x)
print("Error:", error)

plt.figure(figsize=(8, 6))
plt.scatter(x, y, color = "r", marker = "o", s = 30) 
plt.title("Predicted Regression Line", fontsize=16)
y_pred = theta[0] + theta[1]*x 
plt.plot(x, y_pred, color = "b") 
plt.legend(['h(x)', '(x, y)'], loc='best')

plt.xlabel('X', fontsize=14) 
plt.ylabel('Y', fontsize=14)   
plt.show() 
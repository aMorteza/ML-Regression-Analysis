import sys
import scipy.io as sio
from pprint import pprint
import matplotlib.pyplot as plt
import numpy as np
from sympy import *
from scipy.optimize import newton

x = Symbol('x')
y = 2* (x**6) + 3 *(x**4) - 10 *(x**3) + 12 * (x**2) - 60 * x

def dx(y, x, x0):
	f = lambdify(x, y, 'numpy')
	return abs(f(np.array(x0))) 	

def f(x):
    return 2* (x**6) + 3 *(x**4) - 10 *(x**3) + 12 * (x**2) - 60 * x

def df(x):
    return 12 * (x**5) + 12 * (x**3) -30 * (x**3) + 24 * (x**2) - 60 

e = 1e-8
x0 = 1
delta = dx(y, x, x0)
x_pred = []
y_pred = []
x_pred.append(x0)
y_pred.append(f(x0))

while delta > e:
	x0 = x0 - f(x0)/df(x0)
	delta = dx(y, x, x0)
	x_pred.append(x0)
	y_pred.append(f(x0))

print('Root: ', x0)
print('f(x): ', f(x0))

xn = newton(f, 1, df, tol=e)
print('Root with scipy newton: ', xn)
print('f(x) with scipy newton: ', f(xn))

plt.figure(figsize=(10, 8))
plt.title("Newton steps", fontsize=16)
plt.scatter(x_pred, y_pred, color = "b")
plt.xlabel('x', fontsize=14) 
plt.ylabel('f(x)', fontsize=14)   
plt.grid(True)
plt.show() 
import sys
import scipy.io as sio
from pprint import pprint
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

mat_fname = "data_logistic.mat"
mat_dict = sio.loadmat(mat_fname)

samples = mat_dict["logistic_data"]
X1_Y0 = []
X2_Y0 = []
X1_Y1 = []
X2_Y1 = []
for sample in samples:
	if sample[2] == 0:
		X1_Y0.append(sample[0])
		X2_Y0.append(sample[1])
	else:
		X1_Y1.append(sample[0])
		X2_Y1.append(sample[1])

plt.figure(figsize=(8, 6))
plt.title('Training Data', fontsize=16)
plt.scatter(np.array(X1_Y0), np.array(X2_Y0), color='b')
plt.scatter(np.array(X1_Y1), np.array(X2_Y1), color='r')
plt.legend(['Healthy', 'Sick'], loc='best')
plt.xlabel("$X_1$", fontsize=14)
plt.ylabel("$X_2$", fontsize=14)
plt.show()

import scipy.io as sio
import scipy
import numpy as np
import matplotlib.pyplot as plotter
import random
import sklearn.preprocessing
"""load data"""
X = sio.loadmat('data.mat')['X']
X = np.array([np.append(x, np.array([1])) for x in X])
print(np.shape(X))
y = sio.loadmat('data.mat')['y']
"""normalize"""
y = np.array([x[0] for x in y])
y = y/np.linalg.norm(y)
X = sklearn.preprocessing.normalize(np.array(X))
"""set parameters and initialize weight array"""
regularization_param = .07
learning_rate = 1
w = np.zeros(13)
s = scipy.special.expit(np.dot(X, w))
iterator = 0
possible = [x for x in range(6000)] + [x for x in range(6000)]
data_points_x = [10*x for x in range(1200)]
data_points_y = []
while iterator < 12000:
    if(iterator % 10 == 0):
        cost_sum = 0
        for i in range(len(y)):
            cost_sum += y[i]*np.log(s[i]) + (1-y[i])*(np.log(1 - s[i]))
        data_points_y.append(regularization_param*np.power(np.linalg.norm(w), 2) - cost_sum)
    point = random.choice(possible)
    possible.remove(point)
    gradient_loss = -np.dot(np.subtract(y[point],s[point]), X[point])
    gradient_regularization = 2*regularization_param * w
    gradient = gradient_regularization + gradient_loss
    w = np.subtract(w , np.dot(learning_rate/(iterator+1), gradient))
    s = scipy.special.expit(np.dot(X, w))
    iterator += 1
plotter.plot(data_points_x, data_points_y, 'ro')
plotter.show()

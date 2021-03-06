import scipy.io as sio
import scipy
import numpy as np
import matplotlib.pyplot as plotter
import sklearn.preprocessing
"""load data"""
X = sio.loadmat('data.mat')['X']
X = np.array([np.append(x, np.array([1])) for x in X])
print(np.shape(X))
y = sio.loadmat('data.mat')['y']
"""normalize"""
y = np.array([x[0] for x in y])
X = sklearn.preprocessing.normalize(np.array(X))
"""set parameters and initialize weight array"""
regularization_param = .07
learning_rate = .001
w = np.zeros(13)
s = scipy.special.expit(np.dot(X, w))
"""calculate gradient"""
gradient_loss = -np.dot(X.T, np.subtract(y, s).T)
gradient_regularization = 2*regularization_param * w
gradient = np.add(gradient_loss, gradient_regularization)
iterator  = 0
data_points_x = [10*x for x in range(100)]
data_points_y = []
while iterator < 1000:
    if(iterator % 10 == 0):
        cost_sum = 0
        for i in range(len(y)):
            cost_sum += y[i]*np.log(s[i]) + (1-y[i])*(np.log(1 - s[i]))
        data_points_y.append(regularization_param*np.power(np.linalg.norm(w), 2) - cost_sum)
    w = np.subtract(w, np.dot(learning_rate, gradient))
    s = scipy.special.expit(np.dot(X, w))
    gradient_loss = -np.dot(X.T, np.subtract(y, s).T)
    gradient_regularization = 2 * regularization_param * w
    gradient = np.add(gradient_loss, gradient_regularization)
    iterator += 1
plotter.plot(data_points_x, data_points_y, 'ro')
plotter.show()
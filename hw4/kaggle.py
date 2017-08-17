import scipy.io as sio
import scipy
import numpy as np
import csv
import matplotlib.pyplot as plotter
import sklearn.preprocessing
from sklearn.utils import shuffle
"""load data"""
loaded_X = sio.loadmat('data.mat')['X']
loaded_y = sio.loadmat('data.mat')['y']
loaded_X, loaded_y = shuffle(loaded_X, loaded_y, random_state=0)
X = loaded_X[:5000]
X_validation = loaded_X[5000:]
X = np.array([np.append(x, np.array([1])) for x in X])
X_validation = np.array([np.append(x, np.array([1])) for x in X_validation])
y = loaded_y[:5000]
y_validation = loaded_y[5000:]

"""normalize"""
y = np.array([x[0] for x in y])
y_validation = np.array([x[0] for x in y_validation])
X_validation = sklearn.preprocessing.normalize(np.array(X_validation))
X_validation = sklearn.preprocessing.normalize(np.array(X))
"""set parameters and initialize weight array"""
regularization_param = .001
learning_rate = .005
w = np.zeros(13)
s = scipy.special.expit(np.dot(X, w))
"""calculate gradient"""
gradient_loss = -np.dot(X.T, np.subtract(y, s).T)
gradient_regularization = 2*regularization_param * w
gradient = np.add(gradient_loss, gradient_regularization)
iterator  = 0
while iterator < 1000:
    w = np.subtract(w, np.dot(learning_rate, gradient))
    s = scipy.special.expit(np.dot(X, w))
    gradient_loss = -np.dot(X.T, np.subtract(y, s).T)
    gradient_regularization = 2 * regularization_param * w
    gradient = np.add(gradient_loss, gradient_regularization)
    iterator += 1
predictions = scipy.special.expit(np.dot(X_validation, w))
print(predictions)
predictions_classes = []
for x in predictions:
    if (x > 0.5):
        predictions_classes.append(1)
    else:
        predictions_classes.append(0)
correct = 0
for x in range(1000):
    if(predictions_classes[x] == y[x]):
        correct += 1
print(correct/1000)
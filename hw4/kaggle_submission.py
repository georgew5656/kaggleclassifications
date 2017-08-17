import scipy.io as sio
import scipy
import numpy as np
import csv
import matplotlib.pyplot as plotter
import sklearn.preprocessing
from sklearn.utils import shuffle
"""load data"""
test = sio.loadmat('data.mat')['X_test']
loaded_X = sio.loadmat('data.mat')['X']
loaded_y = sio.loadmat('data.mat')['y']
loaded_X, loaded_y = shuffle(loaded_X, loaded_y, random_state=0)
X = loaded_X[:5000]
X = np.array([np.append(x, np.array([1])) for x in X])
test = np.array([np.append(x, np.array([1])) for x in test])
y = loaded_y[:5000]

"""normalize"""
y = np.array([x[0] for x in y])
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
predictions = scipy.special.expit(np.dot(test, w))
print(predictions)
predictions_classes = []
for x in predictions:
    if (x > 0.5):
        predictions_classes.append(1)
    else:
        predictions_classes.append(0)
with open("output_mnist.csv", "w") as f:
    writer = csv.writer(f,lineterminator='\n')
    for index, item in enumerate(predictions_classes):
        writer.writerow([index, item])
import scipy.io as sio
import numpy as np
from sklearn.preprocessing import normalize
from pylab import pcolor, show, colorbar, xticks, yticks
"""import data"""
train = sio.loadmat('hw3_mnist_dist/hw3_mnist_dist/train.mat')['trainX']
classes = {}
means = {}
covariances = {}
"""label classes"""
for x in range(10):
    classes[x] = []
"""group by class"""
for x in train:
    classes[x[-1]].append(x[:(len(x) - 1)])
"""normalize the vectors(int to float)"""
for x in range(10):
    classes[x] = normalize(classes[x])

"""calculate means and covariances"""
for x in range(10):
    means[x] = np.mean(np.array(classes[x]), axis=0)
    covariances[x] = np.cov(np.array(classes[x]).T)
"""view covariance of a class (picked 7)"""
for x in range(10):
    R = covariances[x]
    pcolor(R)
    print("finished")
    colorbar()
    yticks(np.arange(0.5,10.5),range(0,10))
    xticks(np.arange(0.5,10.5),range(0,10))
    show()

import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plotter
from sklearn import svm
"""select validation set"""
cifar_raw = sio.loadmat('hw01_data/cifar/train.mat')['trainX']
np.random.shuffle(cifar_raw)
verification_cifar = cifar_raw[:5000]
training_cifar = cifar_raw[5000:]

pixels = []
labels = []

for x in training_cifar:
    pixels.append(x[:-1])
    labels.append(x[-1])

verification_pixels = []
verification_labels = []
for x in verification_cifar:
    verification_pixels.append(x[:-1])
    verification_labels.append(x[-1])
data_points_x = [100,200,500,1000,2000,5000]
data_points_y = []
classifier = svm.SVC(kernel='linear')
classifier.fit(pixels[:100], labels[:100])
data_points_y.append(classifier.score(verification_pixels, verification_labels))
print(100)
classifier.fit(pixels[100:200], labels[100:200])
data_points_y.append(classifier.score(verification_pixels, verification_labels))
print(200)
classifier.fit(pixels[200:500], labels[200:500])
data_points_y.append(classifier.score(verification_pixels, verification_labels))
print(500)
classifier.fit(pixels[500:1000], labels[500:1000])
data_points_y.append(classifier.score(verification_pixels, verification_labels))
print(1000)
classifier.fit(pixels[1000:2000], labels[1000:2000])
data_points_y.append(classifier.score(verification_pixels, verification_labels))
print(2000)
classifier.fit(pixels[2000:5000], labels[2000:5000])
data_points_y.append(classifier.score(verification_pixels, verification_labels))
print(5000)
"""plot error vs validation/verification set"""
plotter.plot(data_points_x, data_points_y)
plotter.show()

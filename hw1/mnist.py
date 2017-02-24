import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plotter
from sklearn import svm
import csv


"""select validation set"""
mnist_raw = sio.loadmat('hw01_data/mnist/train.mat')['trainX']
np.random.shuffle(mnist_raw)
verification_mnist = mnist_raw[:10000]
training_mnist = mnist_raw[10000:]

pixels = []
labels = []

for x in training_mnist:
    pixels.append(x[:-1])
    labels.append(x[-1])

verification_pixels = []
verification_labels = []
for x in verification_mnist:
    verification_pixels.append(x[:-1])
    verification_labels.append(x[-1])

data_points_x = [100,200,500,1000,2000,5000,10000]
data_points_y = []
classifier = svm.SVC(kernel='linear', C=7*pow(10, -7))
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
classifier.fit(pixels[5000:10000], labels[5000:10000])
data_points_y.append(classifier.score(verification_pixels, verification_labels))
print(10000)
"""for kaggle submission"""
"""classifier.fit(pixels[10000:20000], labels[10000:20000])
print(20000)
classifier.fit(pixels[20000:30000], labels[20000:30000])
print(30000)
classifier.fit(pixels[30000:40000], labels[30000:40000])
print(40000)
classifier.fit(pixels[40000:50000], labels[40000:50000])
print(50000)
classifier.fit(verification_pixels, verification_labels)"""
"""plot error vs verification/validation set"""
plotter.plot(data_points_x, data_points_y)
plotter.show()

mnist_test = sio.loadmat('hw01_data/mnist/test.mat')
prediction = classifier.predict(mnist_test['testX'])
"""
with open("output_mnist.csv", "w") as f:
    writer = csv.writer(f,lineterminator='\n')
    for index, item in enumerate(prediction):
        writer.writerow([index, item])
"""
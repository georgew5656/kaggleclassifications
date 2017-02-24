import scipy.io as sio
import numpy as np
from sklearn import svm


mnist_raw = sio.loadmat('hw01_data/mnist/train.mat')['trainX']
np.random.shuffle(mnist_raw)
verification_mnist = mnist_raw[:10000]
training_mnist = mnist_raw[10000:]

pixels = []
labels = []
"""formatting input for SVM"""
for x in training_mnist:
    pixels.append(x[:-1])
    labels.append(x[-1])
"""formatting verification input for SVM"""
verification_pixels = []
verification_labels = []
for x in verification_mnist:
    verification_pixels.append(x[:-1])
    verification_labels.append(x[-1])
"""first runthrough using c-values between 1 and 10^-10"""
for x in range(10):
    data_points_y = []
    print(data_points_y)
    c_value = pow(10, -1 * x)
    print(c_value)
    classifier = svm.SVC(kernel='linear', C=c_value)
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
    print(data_points_y)
"""second runthrough using c-values between 10^-6 and 10^-7"""
"""for x in range(1, 11):
    data_points_y = []
    print(data_points_y)
    c_value = pow(10, -7) * x
    print(c_value)
    classifier = svm.SVC(kernel='linear', C=c_value)
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
    print(data_points_y)"""
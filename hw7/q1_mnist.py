import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plotter
from scipy.misc import imsave

mnist = sio.loadmat('mnist_data/images.mat')['images']
mnist = np.reshape(mnist, (784, 60000)).T
k_choices = [10]
for k in k_choices:
    mnist = np.random.permutation(mnist)
    means = []
    i = 0
    while len(means) < k:
        new = True
        for x in means:
            if np.array_equal(mnist[i], x):
                new = False
        if new:
            means.append(mnist[i])
        i += 1
    distances = []
    for x in means:
        distances.append(np.linalg.norm(np.subtract(mnist, x), axis=1))
    classes = np.argmin(distances, axis=0)
    changed = 60000
    while changed:
        changed = 0
        #change means
        means = []
        for x in range(k):
            class_elements = [mnist[i] for i in range(60000) if classes[i] == x]
            means.append(np.mean(class_elements, axis=0))
        #change classes
        distances = []
        for x in means:
            distances.append(np.linalg.norm(np.subtract(mnist, x), axis=1))
        new_classes = np.argmin(distances, axis=0)
        changed = 60000 - np.sum(classes == new_classes)
        classes = new_classes
        print(changed)
    means = []
    for x in range(k):
        class_elements = [mnist[i] for i in range(60000) if classes[i] == x]
        means.append(np.mean(class_elements, axis=0))
    for i in range(len(means)):
        imsave(str(k) + str(i) + '.png', np.reshape(means[i], (28, 28)))


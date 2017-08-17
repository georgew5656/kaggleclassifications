import numpy as np
from scipy import special
import scipy.io as sio
import sklearn.preprocessing
import sklearn.metrics
import matplotlib.pyplot as plotter
def train_neural_net(images, labels):
    #normalize the data
    means=np.mean(images, axis=0)
    variances = np.var(images, axis=0)
    #add bias terms
    images = sklearn.preprocessing.scale(images, axis=0, with_mean=True, with_std=True)
    images = np.insert(images, 784, 1, axis=1)
    #vectorize the labels into length 26 vectors
    vectorize = sklearn.preprocessing.OneHotEncoder(sparse=False)
    labels = vectorize.fit_transform(labels)
    #pull out validation data
    validation_images = images[100000:]
    validation_labels = labels[100000:]
    images = images[:100000]
    labels = labels[:100000]
    #initialize weights to random drawn from 0 centered distribution
    V = np.reshape(np.random.normal(0, 1/np.sqrt(785), (200, 785)), (200, 785))
    W = np.reshape(np.random.normal(0, 1/np.sqrt(201), (26, 201)), (26, 201))
    training_rate = .1
    for x in range(1,5):
        training_rate *= 0.9
        from sklearn.utils import shuffle
        epoch = np.array([x for x in range(len(images))])
        epoch = shuffle(epoch)
        while len(epoch)>0:
            batch_indices = epoch[:50]
            epoch = np.delete(epoch, [x for x in range(50)], axis=0)
            batch = np.take(images, batch_indices, axis=0).T
            batch_labels = np.take(labels, batch_indices, axis=0).T
            h = np.tanh(np.dot(V, batch))
            h = np.insert(h, 200, 1, axis=0)
            z = special.expit(np.dot(W, h))
            error = np.subtract(z, batch_labels)
            W_gradient = np.dot(error, h.T)
            W_gradient = np.divide(W_gradient, 50)
            a = np.delete(np.dot(W.T, error), 200, axis=0)
            b = np.subtract(1, np.power(np.tanh(np.dot(V, batch)),2))
            V_gradient = np.dot(np.multiply(a,b), batch.T)
            V_gradient = np.divide(V_gradient, 50)
            W = np.subtract(W,np.multiply(training_rate,W_gradient))
            V = np.subtract(V,np.multiply(training_rate,V_gradient))

    """calculate validation/training accuracy"""
    validation_predictions = neural_net_predict(validation_images, V, W)
    correct_visual = []
    incorrect_visual = []
    for i, x in enumerate(validation_predictions):
        digit = validation_labels[i].argmax(0) + 1
        if digit == x and len(correct_visual) < 5:
            correct_visual.append(validation_images[i][:784])
        if digit != x and len(incorrect_visual) < 5:
            incorrect_visual.append(validation_images[i][:784])
    for x in correct_visual:
        plotter.imshow(np.reshape(x, (28,28)))
        plotter.show()
    for x in incorrect_visual:
        plotter.imshow(np.reshape(x, (28,28)))
        plotter.show()
    return V,W
def neural_net_predict(images, V, W):
    predictions = []
    for i, item in enumerate(images):
        h = np.tanh(np.dot(V, item))
        h = np.insert(h, 200, 1)
        z = special.expit(np.dot(W, h))
        predictions.append(z.argmax(0) + 1)

    return predictions


if __name__ == "__main__":
    from sklearn.utils import shuffle
    import csv
    images = sio.loadmat('hw6_data_dist/letters_data.mat')['train_x']
    labels = sio.loadmat(('hw6_data_dist/letters_data.mat'))['train_y']
    test_data = sio.loadmat('hw6_data_dist/letters_data.mat')['test_x']
    images, labels = shuffle(images, labels)
    V,W = train_neural_net(images, labels)

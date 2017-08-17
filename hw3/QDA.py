import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plotter
import sklearn.preprocessing
"""import data"""
train = sio.loadmat('hw3_mnist_dist/hw3_mnist_dist/train.mat')['trainX'][10000:]
validation = sio.loadmat('hw3_mnist_dist/hw3_mnist_dist/train.mat')['trainX'][:10000]
actual = [x[784] for x in validation]
validation = sklearn.preprocessing.normalize(validation)
def calculate_mean(classes):
    means = {}
    for x in range(10):
        means[x] = np.mean(np.array(classes[x]), axis=0)
    return means
def calculate_covariance(classes, priors):
    covariances = {}
    for x in range(10):
        covariances[x] = np.multiply(np.cov(np.array(classes[x]).T), priors[x])
    return covariances


"""train on different numbers of data points"""
training_sets = [100,200,500,1000,2000,5000,10000,30000,50000]
data_points_x = training_sets
data_points_y = []
for num_datapoints in training_sets:
    np.random.shuffle(train)
    classes = {}
    priors = []
    print(num_datapoints)
    """label classes"""
    for x in range(10):
        classes[x] = []
    """group by class"""
    for x in train[:num_datapoints]:
        classes[x[-1]].append(x[:(len(x) - 1)])
    """calculate priors"""
    for x in range(10):
        priors.append(len(classes[x])/num_datapoints)
    """normalize the vectors(int to float)"""
    for x in range(10):
        classes[x] = sklearn.preprocessing.normalize(classes[x])
    """find parameters"""
    mean_data = calculate_mean(classes)
    covariance_data = calculate_covariance(classes, priors)
    """fix covariance matrix if singular"""
    for x in covariance_data.keys():
        covariance_data[x] = np.add(covariance_data[x], np.diag([0.5 for x in range(len(covariance_data[x]))]))
    """make predictions"""
    inverses = []
    determinants = []
    for x in range(10):
        inverses.append(np.linalg.inv(covariance_data[x]))
        determinants.append(np.linalg.det(covariance_data[x]))
    priors = list(map(lambda x: np.log(x), priors))
    constant_term = []
    for x in range(10):
        constant_term.append(np.log(determinants[x])/2)
    print(constant_term)
    print(priors)
    predictions = []
    for i, item in enumerate(validation):
        class_probabilities = []
        for cls in range(10):
            zeroed = np.subtract(item[:784], mean_data[cls])
            class_probabilities.append(-0.5*np.dot(np.dot(zeroed.T, inverses[x]), zeroed) - constant_term[cls] + priors[cls])
        predictions.append(np.argmax(class_probabilities))
    correct = 0
    for x in range(10000):
        if(predictions[x] == actual[x]):
            correct += 1
    data_points_y.append(correct/10000)
    print(correct/10000)
plotter.plot(data_points_x, data_points_y, 'ro')
plotter.show()



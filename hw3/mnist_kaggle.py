import scipy.io as sio
import numpy as np
import csv
import matplotlib.pyplot as plotter
import sklearn.preprocessing

"""import data"""
train = sio.loadmat('hw3_mnist_dist/hw3_mnist_dist/train.mat')['trainX'][10000:]
validation = sio.loadmat('hw3_mnist_dist/hw3_mnist_dist/test.mat')['testX']
validation = sklearn.preprocessing.normalize(validation)
def calculate_mean(classes):
    means = {}
    for x in range(10):
        means[x] = np.mean(np.array(classes[x]), axis=0)
    return means
def calculate_covariance(classes, priors, number):
    covariances = np.multiply(np.cov(np.array(classes[0]).T), priors[0]*number - 1)
    for x in range(1,10):
        covariances = np.add(covariances, np.multiply(np.cov(np.array(classes[x]).T), priors[x] * number - 1))
    return np.divide(covariances, number - 10)


"""train on different numbers of data points"""
training_sets = [50000]
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
    covariance_data = calculate_covariance(classes, priors, num_datapoints)
    """fix covariance matrix if singular"""
    variance = [x for x in range(784) if covariance_data[x][x] != 0]
    for x in range(10):
        non_zero_variance = []
        for y in range(len(classes[x])):
            temp = [classes[x][y][z] for z in variance]
            non_zero_variance.append(np.append(temp, [np.power(p,2) for p in temp]))
        classes[x] = non_zero_variance

    """apply changes to test vectors"""
    reduced_validation = []
    for x in range(len(validation)):
        updated = [validation[x][z] for z in variance]
        updated = np.append(updated, [np.power(y,2) for y in updated])
        reduced_validation.append(updated)
    """normalize vectors again after fixes"""
    reduced_validation = sklearn.preprocessing.normalize(reduced_validation)
    for x in range(10):
        classes[x] = sklearn.preprocessing.normalize(classes[x])
    mean_data = calculate_mean(classes)
    covariance_data = calculate_covariance(classes, priors, num_datapoints)
    """make predictions"""
    inverse = np.linalg.inv(covariance_data)
    determinant = np.linalg.det(covariance_data)
    priors = list(map(lambda x: np.log(x), priors))
    constant_term = []
    for x in range(10):
        constant_term.append(np.dot(np.dot(mean_data[x].T, inverse),mean_data[x])/2)
    print(constant_term)
    print(priors)
    predictions = []
    for i, item in enumerate(reduced_validation):
        print(i)
        class_probabilities = []
        for cls in range(10):
            class_probabilities.append(np.dot(np.dot(mean_data[cls].T, inverse), item) - constant_term[cls] + priors[cls])
        predictions.append(np.argmax(class_probabilities))

with open("output_mnist.csv", "w") as f:
    writer = csv.writer(f,lineterminator='\n')
    for index, item in enumerate(predictions):
        writer.writerow([index, item])




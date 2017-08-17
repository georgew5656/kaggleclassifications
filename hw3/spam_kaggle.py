import scipy.io as sio
import sklearn.preprocessing
import numpy as np
import csv
everything = list(zip(sio.loadmat('hw3_spam_dist/dist/spam_data.mat')['training_data'], sio.loadmat('hw3_spam_dist/dist/spam_data.mat')['training_labels'][0]))
np.random.shuffle(everything)
validation = sio.loadmat('hw3_spam_dist/dist/spam_data.mat')['test_data']

train = everything
labels =[x[1] for x in train]
train = np.array([x[0] for x in train])
f = sklearn.preprocessing.PolynomialFeatures(interaction_only=True)
train = f.fit_transform(train)
validation = f.fit_transform(validation)
train = sklearn.preprocessing.normalize(train)
validation = sklearn.preprocessing.normalize(validation)
covariance = np.cov(train.T)
variance = [x for x in range(len(covariance)) if covariance[x][x] != 0]
reduced_train = []
for x in train:
    reduced_train.append([x[z] for z in variance])
reduced_validation = []
for y in validation:
    reduced_validation.append([y[z] for z in variance])
reduced_train = np.array(reduced_train)
reduced_train = sklearn.preprocessing.normalize(reduced_train)
reduced_validation = sklearn.preprocessing.normalize(reduced_validation)
covariance = np.cov(reduced_train.T)
covariance = np.add(covariance, np.diag([0.0001 for x in range(len(covariance))]))
inverse = np.linalg.inv(covariance)
zeroes = []
ones = []

for x in range(len(reduced_train)):
    if (labels[x] == 0):
        zeroes.append(reduced_train[x])
    else:
        ones.append(reduced_train[x])
print(len(zeroes))
print(len(ones))
mean_zeroes = np.mean(zeroes, axis=0)
mean_ones = np.mean(ones, axis=0)
prior_zeroes = np.log(len(zeroes)/len(train))
prior_ones = np.log(len(ones)/len(train))
print(prior_zeroes)
print(prior_ones)
constant_zeroes = 0.5*np.dot(np.dot(mean_zeroes.T, inverse), mean_zeroes)
constant_ones = 0.5*np.dot(np.dot(mean_ones.T, inverse), mean_ones)
print(constant_zeroes)
print(constant_ones)
predictions = []
for x in reduced_validation:
    pdf_zeroes = np.dot(np.dot(mean_zeroes.T, inverse), x) - constant_zeroes + prior_zeroes
    pdf_ones = np.dot(np.dot(mean_ones.T, inverse), x) - constant_ones + prior_ones
    if(pdf_zeroes > pdf_ones):
        predictions.append(0)
    else:
        predictions.append(1)
with open("output_spam.csv", "w") as f:
    writer = csv.writer(f,lineterminator='\n')
    for index, item in enumerate(predictions):
        writer.writerow([index, item])


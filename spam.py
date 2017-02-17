import scipy.io as sio
import numpy as np
from sklearn import svm
import csv
import matplotlib.pyplot as plotter
"""select validation set"""
spam_raw = list(zip(sio.loadmat('hw01_data/spam/spam_data.mat')['training_data'], sio.loadmat('hw01_data/spam/spam_data.mat')['training_labels'][0]))
np.random.shuffle(spam_raw)
verification_number = int(len(spam_raw) / 5)
verification_spam = spam_raw[:verification_number]
training_spam = spam_raw[verification_number:]

features = []
labels = []

for x in training_spam:
    features.append(x[0])
    labels.append(x[1])

verification_features = []
verification_labels = []
for x in verification_spam:
    verification_features.append(x[0])
    verification_labels.append(x[1])


data_points_x = [100,200,500,1000,2000, int(len(training_spam))]
data_points_y = []
classifier = svm.SVC(kernel='linear', C=6.6)
classifier.fit(features[:100], labels[:100])
data_points_y.append(classifier.score(verification_features, verification_labels))
print(100)
classifier.fit(features[100:200], labels[100:200])
data_points_y.append(classifier.score(verification_features, verification_labels))
print(200)
classifier.fit(features[200:500], labels[200:500])
data_points_y.append(classifier.score(verification_features, verification_labels))
print(500)
classifier.fit(features[500:1000], labels[500:1000])
data_points_y.append(classifier.score(verification_features, verification_labels))
print(1000)
classifier.fit(features[1000:2000], labels[1000:2000])
data_points_y.append(classifier.score(verification_features, verification_labels))
print(2000)
classifier.fit(features[1000:int(len(training_spam))], labels[1000:int(len(training_spam))])
data_points_y.append(classifier.score(verification_features, verification_labels))
print(len(training_spam))
"""train on verification data for kaggle"""
"""classifier.fit(verification_features, verification_labels)"""
"""plot error vs validation/verification set"""
plotter.plot(data_points_x, data_points_y)
plotter.show()
"""write to output for kaggle submission"""
"""
spam_test = sio.loadmat('hw01_data/spam/spam_data.mat')['test_data']
prediction = classifier.predict(spam_test)
with open("output_spam.csv", "w") as f:
    writer = csv.writer(f,lineterminator='\n')
    for index, item in enumerate(prediction):
        writer.writerow([index, item])"""

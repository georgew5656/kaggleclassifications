import scipy.io as sio
import random
import decision_tree
import random_forest
import csv
import numpy as np
data = sio.loadmat('hw5_spam_dist/dist/spam_data.mat')['training_data']
labels = sio.loadmat('hw5_spam_dist/dist/spam_data.mat')['training_labels'][0]
print(len(data))
print(len(labels))
test = random_forest.RandomForest(8, 25)
roots = test.train(data, labels)

test_data = sio.loadmat('hw5_spam_dist/dist/spam_data.mat')['test_data']
print(len(test_data))
predictions = []
for x in test_data:
    predictions.append(test.predict(x, roots))
print(predictions)
with open("output_spam.csv", "w") as f:
    writer = csv.writer(f,lineterminator='\n')
    for index, item in enumerate(predictions):
        writer.writerow([index, item])
"""best depth= 8"""
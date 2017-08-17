import csv
import decision_tree
import random_forest
import random
data = []
labels = []
with open("hw5_census_dist/cleaned_data.csv") as census_file:
    censusreader = csv.reader(census_file)
    for x in censusreader:
        data.append(list(map(lambda y: int(y), x)))
with open("hw5_census_dist/cleaned_data_labels.csv") as census_file:
    censusreader = csv.reader(census_file)
    for x in censusreader:
        labels.append(int(x[0]))
combined = list(zip(data, labels))
random.shuffle(combined)
data[:], labels[:] = zip(*combined)

"""test on depth 6 width 19 random forest"""
test = random_forest.RandomForest(6, 19)
roots = test.train(data[:25000], labels[:25000])
correct = 0
"""test on validation"""
for x in range(len(data[25000:])):
    prediction = test.predict(data[x], roots)
    if prediction == labels[x]:
        correct += 1
print(correct/len(data[25000:]))
correct = 0
"""test on training"""
for x in range(len(data[:25000])):
    prediction = test.predict(data[x], roots)
    if prediction == labels[x]:
        correct += 1
print(correct/len(data[:25000]))
"""test on depth 6 decision tree"""
test = decision_tree.DecisionTree(6)
root = test.train(data[:25000], labels[:25000])
correct = 0
"""test on validation"""
for x in range(len(data[25000:])):
    prediction = test.predict(data[x], root)
    if prediction == labels[x]:
        correct += 1
print(correct/len(data[25000:]))
correct = 0
"""test on training"""
for x in range(len(data[:25000])):
    prediction = test.predict(data[x], root)
    if prediction == labels[x]:
        correct += 1
print(correct/len(data[:25000]))
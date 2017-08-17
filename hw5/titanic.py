import csv
import decision_tree
import random
import random_forest
data = []
labels = []
with open("hw5_titanic_dist/cleaned_data.csv") as census_file:
    censusreader = csv.reader(census_file)
    for x in censusreader:
        data.append(list(map(lambda y: int(y), x)))
with open("hw5_titanic_dist/cleaned_data_labels.csv") as census_file:
    censusreader = csv.reader(census_file)
    for x in censusreader:
        labels.append(int(x[0]))
"""test with depth 6 width 19 random forest"""
test = random_forest.RandomForest(6, 19)
roots = test.train(data[:800], labels[:800])
correct = 0
"""test on validation"""
for x in range(len(data[800:])):
    prediction = test.predict(data[x], roots)
    if prediction == labels[x]:
        correct += 1
print(correct/len(data[800:]))
correct = 0
"""test on training"""
for x in range(len(data[:800])):
    prediction = test.predict(data[x], roots)
    if prediction == labels[x]:
        correct += 1
print(correct/len(data[:800]))
"""test with depth 6 decition tree"""
test = decision_tree.DecisionTree(6)
root = test.train(data[:800], labels[:800])
correct = 0
"""test on validation"""
for x in range(len(data[800:])):
    prediction = test.predict(data[x], root)
    if prediction == labels[x]:
        correct += 1
print(correct/len(data[800:]))
correct = 0
"""test on training"""
for x in range(len(data[:800])):
    prediction = test.predict(data[x], root)
    if prediction == labels[x]:
        correct += 1
print(correct/len(data[:800]))

import decision_tree
import csv
import random
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

test = decision_tree.DecisionTree(2)
root = test.train(data, labels)
print(root.split_rule)
print(root.left.split_rule)
print(root.right.split_rule)
print(root.left.left.label)
print(root.left.right.label)
print(root.right.left.label)
print(root.right.right.label)
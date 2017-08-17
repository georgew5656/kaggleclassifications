import scipy.io as sio
import random
import decision_tree
import random_forest
data = sio.loadmat('hw5_spam_dist/dist/spam_data.mat')['training_data']
labels = sio.loadmat('hw5_spam_dist/dist/spam_data.mat')['training_labels'][0]
combined = list(zip(data, labels))
random.shuffle(combined)
data[:], labels[:] = zip(*combined)
"""depth 8 decision tree"""
test = decision_tree.DecisionTree(8)
root = test.train(data[:19000], labels[:19000])
correct = 0


"""test on validation"""
for x in range(len(data[19000:])):
    prediction = test.predict(data[x], root)
    if prediction == labels[x]:
        correct += 1
print(correct/len(data[19000:]))
correct = 0
"""test on training"""
for x in range(len(data[:19000])):
    prediction = test.predict(data[x], root)
    if prediction == labels[x]:
        correct += 1
print(correct/len(data[:19000]))
"""detph 8 width 19 random forest"""
test = random_forest.RandomForest(8, 19)
roots = test.train(data[:19000], labels[:19000])
correct = 0
"""test on validation"""
for x in range(len(data[19000:])):
    prediction = test.predict(data[x], roots)
    if prediction == labels[x]:
        correct += 1
print(correct/len(data[19000:]))
correct = 0
"""test on training"""
for x in range(len(data[:19000])):
    prediction = test.predict(data[x], roots)
    if prediction == labels[x]:
        correct += 1
print(correct/len(data[:19000]))

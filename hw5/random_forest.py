import decision_tree
import numpy as np
from sklearn.utils import shuffle
class RandomForest:
    def __init__(self, maxDepth, trees):
        self.maxDepth = maxDepth
        self.trees = trees
    def train(self, data, labels):
        decision_trees = []
        test = decision_tree.DecisionTree(self.maxDepth)
        segments = len(data)//2
        m = np.sqrt(len(data[0]))
        for x in range(self.trees):
            data, labels = shuffle(data, labels)
            decision_trees.append(test.train(data[:segments], labels[:segments], m=int(m)))
        return decision_trees
    def predict(self, data, roots):
        predictions = [0,0]
        test = decision_tree.DecisionTree(self.maxDepth)
        for x in roots:
            predictions[test.predict(data, x)] += 1
        if predictions[0] > predictions[1]:
            return 0
        else:
            return 1
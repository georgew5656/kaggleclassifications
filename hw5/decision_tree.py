import census_clean
import numpy as np
import operator
import csv
import random

class Node:
    def __init__(self, leaf=False, label=None, split_rule=None, left=None, right=None):
        self.leaf = leaf
        self.split_rule = split_rule
        self.left = left
        self.right = right
        self.label = label
class DecisionTree:
    def __init__(self, maxDepth):
        self.maxDepth = maxDepth
    def impurity(self, left_label_hist, right_label_hist):
        left_total = left_label_hist[0] + left_label_hist[1]
        right_total = right_label_hist[0] + right_label_hist[1]
        entropy_left_zero = 0
        entropy_left_one = 0
        entropy_right_zero = 0
        entropy_right_one = 0
        if (left_label_hist[0] != 0):
            entropy_left_zero = (left_label_hist[0] / left_total) * np.log2(left_label_hist[0] / left_total)
        if (left_label_hist[1] != 0):
            entropy_left_one = (left_label_hist[1] / left_total) * np.log2(left_label_hist[1] / left_total)
        if (right_label_hist[0] != 0):
            entropy_right_zero = (right_label_hist[0] / right_total) * np.log2(right_label_hist[0] / right_total)
        if (right_label_hist[1] != 0):
            entropy_right_one = (right_label_hist[1] / right_total) * np.log2(right_label_hist[1] / right_total)
        entropy_left = -(entropy_left_zero + entropy_left_one)
        entropy_right = -(entropy_right_zero + entropy_right_one)
        return (left_total * entropy_left + right_total * entropy_right) / (left_total + right_total)
    def segmenter(self, data, labels, m):
        from sklearn.utils import shuffle
        feature_list = list(range(len(data[0])))
        if m:
            feature_list = shuffle(feature_list)
            feature_list = feature_list[:m]
        """create histogram of possible values for each feature"""
        possible_splits = {}
        for feature in feature_list:
            feature_possible_values = {}
            for index, point in enumerate(data):
                if(point[feature] in feature_possible_values.keys()):
                    feature_possible_values[point[feature]][labels[index]] += 1
                else:
                    feature_possible_values[point[feature]] = [0, 0]
                    feature_possible_values[point[feature]][labels[index]] += 1
            feature_possible_values = list(feature_possible_values.items())
            feature_possible_values.sort()
            possible_splits[feature] = feature_possible_values
        """call impurity to find best feature/splitting value"""
        best_split = [(0,0), 100]
        total_points = [0, 0]
        for label in labels:
            total_points[label] += 1
        for feature in feature_list:
            left = [0, 0]
            right = list(total_points)
            for index in range(len(possible_splits[feature][:-1])):
                split_value = (possible_splits[feature][index][0] + possible_splits[feature][index+1][0])/2
                left[0] += possible_splits[feature][index][1][0]
                left[1] += possible_splits[feature][index][1][1]
                right[0] -= possible_splits[feature][index][1][0]
                right[1] -= possible_splits[feature][index][1][1]
                entropy = self.impurity(left, right)
                if (entropy < best_split[1]):
                    best_split = [(feature, split_value), entropy]
        return best_split[0]
    def train(self, data, labels, depth=1, m=0):
        totals = [0 ,0]
        for x in labels:
            totals[x] += 1
        if(depth > self.maxDepth):
            if (totals[1] > totals[0]):
                return Node(leaf=True, label=1)
            else:
                return Node(leaf=True, label=0)
        if not totals[0]:
            return Node(leaf=True, label=1)
        if not totals[1]:
            return Node(leaf=True, label=0)
        rule = self.segmenter(data, labels, m)
        left_data = []
        left_labels = []
        right_data = []
        right_labels = []
        for index, point in enumerate(data):
            if (point[rule[0]] > rule[1]):
                right_data.append(point)
                right_labels.append(labels[index])
            else:
                left_data.append(point)
                left_labels.append(labels[index])
        return Node(split_rule=rule, left=self.train(left_data, left_labels, depth+ 1), right=self.train(right_data, right_labels, depth+1))
    def predict(self, datapoint, root):
        while not root.leaf:
            feature = root.split_rule[0]
            value = root.split_rule[1]
            """check which side to go to"""
            if(datapoint[feature] > value):
                root = root.right
            else:
                root = root.left
        return root.label
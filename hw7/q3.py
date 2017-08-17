import scipy.io as sio
import numpy as np
import sklearn.preprocessing
import matplotlib.pyplot as plotter
with open('joke_data/validation.txt') as f:
    validation_jokes = f.readlines()
validation = [x.split(',') for x in validation_jokes]
people = sio.loadmat('joke_data/joke_train.mat')['train']
people = np.nan_to_num(people)
u,s,v = np.linalg.svd(people, False)
d_values = [2,5,10,20]
for d in d_values:
    new_s = np.append(s[:d], np.zeros(len(s) - d))
    new_s = np.sqrt(new_s)
    transformed_people = np.dot(u, np.diag(new_s)).T[:d].T
    transformed_jokes = np.dot(v.T, np.diag(new_s)).T[:d].T
    correct = 0
    for x in validation:
        output = np.dot(transformed_people[int(x[0]) - 1], transformed_jokes[int(x[1]) - 1])
        if output >= 0:
            if int(x[2]) == 1:
                correct += 1
        else:
            if int(x[2]) != 1:
                correct += 1
    print(correct/len(validation))
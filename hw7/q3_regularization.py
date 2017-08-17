import scipy.io as sio
import numpy as np
import sklearn.preprocessing
import matplotlib.pyplot as plotter
with open('joke_data/validation.txt') as f:
    validation_jokes = f.readlines()
validation_jokes = [x.strip() for x in validation_jokes]
validation = [x.split(',') for x in validation_jokes]

people = sio.loadmat('joke_data/joke_train.mat')['train']
reg_constant = 300
d_values = [8,9,10]
for d in d_values:
    transformed_people = np.random.normal(0, 1, (len(people), d))
    transformed_jokes = np.random.normal(0, 1, (100, d))
    #minimize u's (people)
    for n in range(20):
        print(n)
        for i in range(len(transformed_people)):
            nan_list = np.argwhere(~np.isnan(people[i]))
            nan = [x[0] for x in nan_list]
            b = 2*np.dot(np.take(people[i], nan), np.take(transformed_jokes, nan, axis=0))
            A = 2*np.dot(np.take(transformed_jokes.T,nan, axis=1), np.take(transformed_jokes, nan, axis=0)) + np.diag(np.array([2*reg_constant for x in range(d)]))
            transformed_people[i] = np.linalg.solve(A, b)
        for i in range(len(transformed_jokes)):
            nan_list = np.argwhere(~np.isnan(people.T[i]))
            nan = [x[0] for x in nan_list]
            b = 2*np.dot(np.take(people.T[i], nan), np.take(transformed_people,nan, axis=0))
            A = 2*np.dot(np.take(transformed_people.T,nan,axis=1), np.take(transformed_people, nan, axis=0)) + np.diag(np.array([2*reg_constant for x in range(d)]))
            transformed_jokes[i] = np.linalg.solve(A,b)
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
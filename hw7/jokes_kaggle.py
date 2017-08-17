import scipy.io as sio
import numpy as np
import sklearn.preprocessing
import csv
import matplotlib.pyplot as plotter
people = sio.loadmat('joke_data/joke_train.mat')['train']
reg_constant = 300
d_values = [9]
with open('joke_data/query.txt') as f:
    test_jokes = f.readlines()
test_jokes = [x.strip() for x in test_jokes]
test = [x.split(',') for x in test_jokes]
print(len(test))
for d in d_values:
    transformed_people = np.random.normal(0, 1, (len(people), d))
    transformed_jokes = np.random.normal(0, 1, (100, d))
    #minimize u's (people)
    for n in range(40):
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
predictions = []
ids = []
for x in test:
    output = np.dot(transformed_people[int(x[1]) - 1], transformed_jokes[int(x[2]) - 1])
    ids.append(x[0])
    if output >= 0:
        predictions.append(1)
    else:
        predictions.append(0)
with open("output_jokess.csv", "w") as f:
    writer = csv.writer(f, lineterminator='\n')
    for index, item in enumerate(predictions):
        writer.writerow([ids[index], item])
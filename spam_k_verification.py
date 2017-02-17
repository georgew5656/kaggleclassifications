import scipy.io as sio
import numpy as np
from sklearn import svm
k=5
spam_raw = list(zip(sio.loadmat('hw01_data/spam/spam_data.mat')['training_data'], sio.loadmat('hw01_data/spam/spam_data.mat')['training_labels'][0]))
np.random.shuffle(spam_raw)
verification_number = int(len(spam_raw) / 5)
verification_sets = []
for n in range(k):
    verification_sets.append(spam_raw[n*verification_number:(n+1)*verification_number])
    features = list(map(lambda x: x[0], verification_sets[n]))
    labels = list(map(lambda x: x[1], verification_sets[n]))
    verification_sets[n] = (features, labels)
"""first round of c-values"""
for x in range(1, 11):
    accuracy_scores = []
    for n in range(k):
        classifier = svm.SVC(kernel='linear', C=x)
        for m in range(k):
            if(m != n):
                classifier.fit(verification_sets[m][0], verification_sets[m][1])
        accuracy_scores.append(classifier.score(verification_sets[n][0], verification_sets[n][1]))
    print(sum(accuracy_scores)/k)
"""second round of c-values"""
"""
for x in range(1,11):
    accuracy_scores = []
    for n in range(k):
        classifier = svm.SVC(kernel='linear', C=(6 + x/10))
        for m in range(k):
            if(m != n):
                classifier.fit(verification_sets[m][0], verification_sets[m][1])
        accuracy_scores.append(classifier.score(verification_sets[n][0], verification_sets[n][1]))
    print(sum(accuracy_scores)/k)
"""
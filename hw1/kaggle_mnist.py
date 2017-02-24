from os import listdir
import re
spam_list = []
ham_list = []
for x in listdir('hw01_data/spam/spam'):
    spam_list.append(x)

for x in listdir('hw01_data/spam/ham'):
    ham_list.append(x)

spam_frequency = {}
for x in spam_list:
    f = open('hw01_data/spam/spam/' + x, 'r', errors='ignore')
    email = f.read()
    """email = re.sub('[^0-9a-zA-Z]+', ' ', email)"""
    words = email.split()
    for x in words:
        if (x in spam_frequency):
            spam_frequency[x] += 1
        else:
            spam_frequency[x] = 1
spam_frequencies_sorted = sorted(spam_frequency.items(), key=lambda x: x[1])
ham_frequency = {}
for x in ham_list:
    f = open('hw01_data/spam/ham/' + x, 'r', errors='ignore')
    email = f.read()
    """email = re.sub('[^0-9a-zA-Z]+', ' ', email)"""
    words = email.split()
    for x in words:
        if (x in ham_frequency):
            ham_frequency[x] += 1
        else:
            ham_frequency[x] = 1
ham_frequencies_sorted = sorted(ham_frequency.items(), key=lambda x: x[1])
spam_frequencies_sorted = sorted(spam_frequency.items(), key=lambda x: x[1])
spammy = []
for x in ham_frequency:
    if x in spam_frequency:
        spammy.append((x,ham_frequency[x] - spam_frequency[x]))
spammy = sorted(spammy, key=lambda x: x[1])
print(spammy[-25:])
print(spammy[:25])

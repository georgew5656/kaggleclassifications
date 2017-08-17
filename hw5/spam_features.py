import os
import string
import operator
spam_path = "hw5_spam_dist/dist/spam/"
ham_path = "hw5_spam_dist/dist/ham/"
spam_emails = []
ham_emails = []
print("asdf")
for spam_email in os.listdir(spam_path):
    print(spam_email)
    f = open(spam_path + spam_email, 'r', errors='ignore')
    spam_emails.append(f.read())
for ham_email in os.listdir(ham_path):
    print(ham_email)
    f = open(ham_path + ham_email, 'r', errors='ignore')
    ham_emails.append(f.read())
for x in range(len(spam_emails)):
    print(x)
    spam_emails[x] = spam_emails[x].split()
for x in range(len(ham_emails)):
    print(x)
    ham_emails[x] = ham_emails[x].split()
spam_words = {}
ham_words = {}
for x in spam_emails:
    for y in x:
        if y in spam_words:
            spam_words[y] += 1
        else:
            spam_words[y] = 1
for x in ham_emails:
    for y in x:
        if y in ham_words:
            ham_words[y] += 1
        else:
            ham_words[y] = 1

difference = {}
divide = {}
both = set(spam_words.keys()) & set(ham_words.keys())
spam = set(spam_words.keys()) - set(ham_words.keys())
ham = set(ham_words.keys()) - set(spam_words.keys())
for z in both:
    difference[z] = spam_words[z] - ham_words[z]
    divide[z] = (spam_words[z] / ham_words[z], spam_words[z] - ham_words[z])





sorted_differences = sorted(difference.items(), key=operator.itemgetter(1))
sorted_divide = sorted(divide.items(), key=operator.itemgetter(1))
print([x for x in sorted_divide if x[1][1] < -5000])
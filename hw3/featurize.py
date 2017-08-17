'''
**************** PLEASE READ ***************

Script that reads in spam and ham messages and converts each training example
into a feature vector

Code intended for UC Berkeley course CS 189/289A: Machine Learning

Requirements:
-scipy ('pip install scipy')

To add your own features, create a function that takes in the raw text and
word frequency dictionary and outputs a int or float. Then add your feature
in the function 'def generate_feature_vector'

The output of your file will be a .mat file. The data will be accessible using
the following keys:
    -'training_data'
    -'training_labels'
    -'test_data'

'''

from collections import defaultdict
import glob
import re
import scipy.io

NUM_TEST_EXAMPLES = 10000

BASE_DIR = './'
SPAM_DIR = 'spam/'
HAM_DIR = 'ham/'
TEST_DIR = 'test/'

# ************* Features *************

# Features that look for certain words
def freq_pain_feature(text, freq):
    return float(freq['pain'])

def freq_money_feature(text, freq):
    return float(freq['money'])

def freq_drug_feature(text, freq):
    return float(freq['drug'])

def freq_spam_feature(text, freq):
    return float(freq['spam'])

def freq_prescription_feature(text, freq):
    return float(freq['prescription'])

def freq_creative_feature(text, freq):
    return float(freq['creative'])

def freq_height_feature(text, freq):
    return float(freq['height'])

def freq_featured_feature(text, freq):
    return float(freq['featured'])

def freq_differ_feature(text, freq):
    return float(freq['differ'])

def freq_width_feature(text, freq):
    return float(freq['width'])

def freq_energy_feature(text, freq):
    return float(freq['energy'])


def freq_volumes_feature(text, freq):
    return float(freq['volumes'])

def freq_revision_feature(text, freq):
    return float(freq['revision'])

def freq_path_feature(text, freq):
    return float(freq['path'])

def freq_meter_feature(text, freq):
    return float(freq['meter'])

def freq_memo_feature(text, freq):
    return float(freq['memo'])

def freq_planning_feature(text, freq):
    return float(freq['planning'])


# Features that look for certain characters

def freq_dollar_feature(text, freq):
    return text.count('$')


def freq_exclamation_feature(text, freq):
    return text.count('!')


def freq_and_feature(text, freq):
    return text.count('&')


# --------- Add your own feature methods ----------
def php(text, freq):
    return float(freq['php'])

def meds(text, freq):
    return float(meds['meds'])

def oi(text, freq):
    return float(freq['oi'])

def macromedia(text, freq):
    return float(freq['macromedia'])

def softwares(text, freq):
    return float(freq['softwares'])

def oem(text, freq):
    return float(freq['oem'])

def enron(text, freq):
    return float(freq['enron'])

def ect(text, freq):
    return float(freq['ect'])

def hpl(text, freq):
    return float(freq['hpl'])

def crenshaw(text, freq):
    return float(freq['crenshaw'])

def stinson(text, freq):
    return float(freq['stinson'])

def ene(text, freq):
    return float(freq['ene'])

def louise(text, freq):
    return float(freq['louise'])

def ubs(text, freq):
    return float(freq['ubs'])

def sevenonethree(text, freq):
    return float(freq['713'])

def daren(text, freq):
    return float(freq['daren'])

def vince(text, freq):
    return float(freq['vince'])

def schedules(text, freq):
    return float(freq['schedules'])

def reuters(text, freq):
    return float(freq['reuters'])

def ratings(text, freq):
    return float(freq['ratings'])

def forwarded(text, freq):
    return float(freq['forwarded'])

def corp(text, freq):
    return float(freq['corp'])

def fyi(text, freq):
    return float(freq['fyi'])

def cc(text, freq):
    return float(freq['cc'])

def houston(text, freq):
    return float(freq['houston'])

def eightfivethree(text, freq):
    return float(freq['853'])

def hou(text, freq):
    return float(freq['hou'])

def twozerozeroone(text, freq):
    return float(freq['2001'])

def pm(text, freq):
    return float(freq['pm'])

def meeting(text, freq):
    return float(freq['meeting'])

def bigger(text, freq):
    return text.count('>')

def underline(text, freq):
    return text.count('_')

def bar(text, freq):
    return text.count('|')

def corp(text, freq):
    return float(freq['corp'])
def said(text, freq):
    return float(freq['said'])
def power(text, freq):
    return float(freq['power'])
def statements(text, freq):
    return float(freq['statements'])
def de(text, freq):
    return float(freq['de'])
# Generates a feature vector
def generate_feature_vector(text, freq):
    feature = []
    feature.append(freq_pain_feature(text, freq))
    feature.append(freq_money_feature(text, freq))
    feature.append(freq_drug_feature(text, freq))
    feature.append(freq_spam_feature(text, freq))
    feature.append(freq_prescription_feature(text, freq))
    feature.append(freq_creative_feature(text, freq))
    feature.append(freq_height_feature(text, freq))
    feature.append(freq_featured_feature(text, freq))
    feature.append(freq_differ_feature(text, freq))
    feature.append(freq_width_feature(text, freq))
    feature.append(freq_energy_feature(text, freq))
    feature.append(freq_volumes_feature(text, freq))
    feature.append(freq_revision_feature(text, freq))
    feature.append(freq_path_feature(text, freq))
    feature.append(freq_meter_feature(text, freq))
    feature.append(freq_memo_feature(text, freq))
    feature.append(freq_dollar_feature(text, freq))
    feature.append(freq_exclamation_feature(text, freq))
    feature.append(php(text, freq))
    feature.append(macromedia(text, freq))
    feature.append(oem(text, freq))
    feature.append(softwares(text, freq))
    feature.append(oi(text, freq))
    feature.append(ene(text, freq))
    feature.append(sevenonethree(text, freq))
    feature.append(ubs(text, freq))
    feature.append(louise(text, freq))
    feature.append(stinson(text, freq))
    feature.append(crenshaw(text, freq))
    feature.append(hpl(text, freq))
    feature.append(ect(text, freq))
    feature.append(enron(text, freq))
    feature.append(vince(text, freq))
    feature.append(schedules(text, freq))
    feature.append(forwarded(text, freq))
    feature.append(fyi(text, freq))
    feature.append(corp(text, freq))
    feature.append(reuters(text, freq))
    feature.append(ratings(text, freq))
    feature.append(cc(text, freq))
    feature.append(houston(text, freq))
    feature.append(hou(text, freq))
    feature.append(twozerozeroone(text, freq))
    feature.append(pm(text, freq))
    feature.append(meeting(text, freq))
    feature.append(bigger(text, freq))
    feature.append(underline(text, freq))
    feature.append(bar(text, freq))
    feature.append(corp(text, freq))
    feature.append(said(text, freq))
    feature.append(power(text, freq))
    feature.append(de(text, freq))
    feature.append(statements(text, freq))
    # --------- Add your own features here ---------
    # Make sure type is int or float

    return feature

# This method generates a design matrix with a list of filenames
# Each file is a single training example
def generate_design_matrix(filenames):
    design_matrix = []
    for filename in filenames:
        with open(filename, encoding='utf-8', errors='ignore') as f:
            text = f.read() # Read in text from file
            text = text.replace('\r\n', ' ') # Remove newline character
            words = re.findall(r'\w+', text)
            word_freq = defaultdict(int) # Frequency of all words
            for word in words:
                word_freq[word] += 1

            # Create a feature vector
            feature_vector = generate_feature_vector(text, word_freq)
            design_matrix.append(feature_vector)
    return design_matrix

# ************** Script starts here **************
# DO NOT MODIFY ANYTHING BELOW

spam_filenames = glob.glob(BASE_DIR + SPAM_DIR + '*.txt')
spam_design_matrix = generate_design_matrix(spam_filenames)
ham_filenames = glob.glob(BASE_DIR + HAM_DIR + '*.txt')
ham_design_matrix = generate_design_matrix(ham_filenames)
# Important: the test_filenames must be in numerical order as that is the
# order we will be evaluating your classifier
test_filenames = [BASE_DIR + TEST_DIR + str(x) + '.txt' for x in range(NUM_TEST_EXAMPLES)]
test_design_matrix = generate_design_matrix(test_filenames)

X = spam_design_matrix + ham_design_matrix
Y = [1]*len(spam_design_matrix) + [0]*len(ham_design_matrix)

file_dict = {}
file_dict['training_data'] = X
file_dict['training_labels'] = Y
file_dict['test_data'] = test_design_matrix
scipy.io.savemat('spam_data.mat', file_dict, do_compression=True)


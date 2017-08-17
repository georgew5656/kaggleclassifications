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

def freq_php_feature(text, freq):
    return float(freq['php'])

def freq_macromedia_feature(text, freq):
    return float(freq['macromedia'])

def freq_oi_feature(text, freq):
    return float(freq['oi'])

def freq_meds_feature(text, freq):
    return float(freq['meds'])

def freq_font_feature(text, freq):
    return float(freq['font'])

def freq_oo_feature(text, freq):
    return float(freq['oo'])

def freq_adobe_feature(text, freq):
    return float(freq['adobe'])

def freq_twofive_feature(text, freq):
    return float(freq['2001'])

def freq_statements_feature(text, freq):
    return float(freq['statements'])

def freq_de_feature(text, freq):
    return float(freq['de'])
def freq_account_feature(text, freq):
    return float(freq['account'])
def freq_money_feature(text, freq):
    return float(freq['money'])
def freq_click_feature(text, freq):
    return float(freq['click'])
def freq_http_feature(text, freq):
    return float(freq['http'])
def freq_ect_feature(text, freq):
    return float(freq['ect'])
def freq_hou_feature(text, freq):
    return float(freq['hou'])
def freq_on_feature(text, freq):
    return float(freq['on'])
def freq_vince_feature(text, freq):
    return float(freq['vince'])

def freq_pm_feature(text, freq):
    return float(freq['pm'])

def freq_said_feature(text, freq):
    return float(freq['said'])

def freq_cc_feature(text, freq):
    return float(freq['cc'])
def freq_energy_feature(text, freq):
    return float(freq['energy'])
def freq_subject_feature(text, freq):
    return float(freq['subject'])
def freq_would_feature(text, freq):
    return float(freq['would'])
# Features that look for certain characters
def freq_semicolon_feature(text, freq):
    return text.count(';')

def freq_exclamation_feature(text, freq):
    return text.count('!')

def freq_para_feature(text, freq):
    return text.count('(')
def freq_bar_feature(text, freq):
    return text.count('|')
def freq_bigger_feature(text, freq):
    return text.count('>')
def freq_at_feature(text, freq):
    return text.count('@')

def freq_and_feature(text, freq):
    return text.count('&')

# --------- Add your own feature methods ----------
def example_feature(text, freq):
    return int('example' in text)

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
    feature.append(freq_energy_feature(text, freq))
    feature.append(freq_volumes_feature(text, freq))
    feature.append(freq_revision_feature(text, freq))
    feature.append(freq_path_feature(text, freq))
    feature.append(freq_meter_feature(text, freq))
    feature.append(freq_memo_feature(text, freq))
    feature.append(freq_planning_feature(text, freq))
    feature.append(freq_semicolon_feature(text, freq))
    feature.append(freq_exclamation_feature(text, freq))
    feature.append(freq_para_feature(text, freq))
    feature.append(freq_php_feature(text, freq))
    feature.append(freq_meds_feature(text, freq))
    feature.append(freq_oi_feature(text, freq))
    feature.append(freq_twofive_feature(text, freq))
    feature.append(freq_adobe_feature(text, freq))
    feature.append(freq_oo_feature(text, freq))
    feature.append(freq_font_feature(text, freq))
    feature.append(freq_bar_feature(text, freq))
    feature.append(freq_account_feature(text, freq))
    feature.append(freq_money_feature(text, freq))
    feature.append(freq_click_feature(text, freq))
    feature.append(freq_http_feature(text, freq))
    feature.append(freq_bigger_feature(text, freq))
    feature.append(freq_at_feature(text, freq))
    feature.append(freq_vince_feature(text, freq))
    feature.append(freq_pm_feature(text, freq))
    feature.append(freq_cc_feature(text, freq))
    feature.append(freq_said_feature(text, freq))
    feature.append(freq_energy_feature(text, freq))
    feature.append(freq_subject_feature(text, freq))
    feature.append(freq_would_feature(text, freq))
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


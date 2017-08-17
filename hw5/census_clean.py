import csv
import sklearn.feature_extraction
import numpy as np
import operator
def vectorize(cleaned_datapoints):
    numerical = ['age', 'fnlwgt', 'education-num', 'capital-gain', 'capital-loss','hours-per-week']
    #put non-cateogrical features into vector and remove labels
    datapoints_vectorized = []
    labels = []
    for x in cleaned_datapoints:
        datapoint_vectorized = []
        for feature in numerical:
            datapoint_vectorized.append(float(x[feature]))
            x.pop(feature, None)
        datapoints_vectorized.append(datapoint_vectorized)
        labels.append(int(x['label']))
        x.pop('label', None)
    #vectorize categorical features
    v = sklearn.feature_extraction.DictVectorizer(sparse=False)
    cleaned_datapoints = v.fit_transform(cleaned_datapoints)

    for i in range(len(cleaned_datapoints)):
        datapoints_vectorized[i] = np.concatenate((datapoints_vectorized[i], cleaned_datapoints[i]))
    return(datapoints_vectorized, labels)
def fill_misisng_with_mean():
    datapoints = []
    categorical = ['workclass', 'education','marital-status','occupation','relationship', 'race', 'sex', 'native-country']
    numerical = ['age', 'fnlwgt', 'education-num', 'capital-gain', 'capital-loss','hours-per-week']
    with open("hw5_census_dist/train_data.csv") as census_file:
        censusreader = csv.DictReader(census_file)
        for x in censusreader:
            datapoints.append(x)
    datapoints_columns = {i:[x[i] for x in datapoints] for i in categorical+numerical}
    means = {}
    for header in categorical + numerical:
        if (header in numerical):
            total = 0
            for point in datapoints_columns[header]:
                if (point != '?'):
                    total += float(point)
            means[header] = total/len(datapoints_columns[header])
        else:
            occurances = {}
            for point in datapoints_columns[header]:
                if (point != '?'):
                    if point in occurances:
                        occurances[point] += 1
                    else:
                        occurances[point] = 1
            max_key = max(occurances.items(), key=operator.itemgetter(1))[0]
            means[header] = max_key
    for data_point in datapoints:
        for key in data_point:
            if data_point[key]=='?':
                data_point[key] = means[key]

    return datapoints
def write_to_file():
    cleaned = fill_misisng_with_mean()
    vectorized = vectorize(cleaned)
    with open("hw5_census_dist/cleaned_data.csv", 'w', newline='') as census_file:
        writer = csv.writer(census_file)
        for x in vectorized[0]:
            writer.writerow(list(map(lambda y: int(y), x)))
    with open("hw5_census_dist/cleaned_data_labels.csv", 'w', newline='') as census_file:
        writer = csv.writer(census_file)
        for x in vectorized[1]:
            writer.writerow([int(x)])
    return
write_to_file()
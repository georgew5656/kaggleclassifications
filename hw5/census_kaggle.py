import csv
import decision_tree
import random
import random_forest
data = []
labels = []
test_data = []
with open("hw5_census_dist/cleaned_data.csv") as census_file:
    censusreader = csv.reader(census_file)
    for x in censusreader:
        data.append(list(map(lambda y: int(y), x)))
with open("hw5_census_dist/cleaned_data_labels.csv") as census_file:
    censusreader = csv.reader(census_file)
    for x in censusreader:
        labels.append(int(x[0]))
with open("hw5_census_dist/cleaned_test_data.csv") as census_file:
    censusreader = csv.reader(census_file)
    for x in censusreader:
        test_data.append(list(map(lambda y: int(y), x)))
test = random_forest.RandomForest(6, 19)
root = test.train(data, labels)
predictions = []
for x in test_data:
    predictions.append(test.predict(x, root))

with open("output_census.csv", "w") as f:
    writer = csv.writer(f,lineterminator='\n')
    for index, item in enumerate(predictions):
        writer.writerow([index+1, item])
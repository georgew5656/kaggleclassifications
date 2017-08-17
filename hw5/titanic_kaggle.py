import csv
import decision_tree
import random
import random_forest
data = []
labels = []
test_data = []
with open("hw5_titanic_dist/cleaned_data.csv") as census_file:
    censusreader = csv.reader(census_file)
    for x in censusreader:
        data.append(list(map(lambda y: int(y), x)))
with open("hw5_titanic_dist/cleaned_data_labels.csv") as census_file:
    censusreader = csv.reader(census_file)
    for x in censusreader:
        labels.append(int(x[0]))
with open("hw5_titanic_dist/cleaned_data_test.csv") as census_file:
    censusreader = csv.reader(census_file)
    for x in censusreader:
        test_data.append(list(map(lambda y: int(y), x)))
print(len(data[0]))
print(len(test_data[0]))
test = random_forest.RandomForest(6,29)
root = test.train(data, labels)
predictions = []
for x in test_data:
    predictions.append(test.predict(x, root))
print(predictions)

with open("output_titanic.csv", "w") as f:
    writer = csv.writer(f,lineterminator='\n')
    for index, item in enumerate(predictions):
        writer.writerow([index+1, item])
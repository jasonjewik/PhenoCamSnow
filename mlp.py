from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score
import csv


def parse_row(row):
    fname = row[0]
    ratios = []
    for x in row[1:]:
        x = x.replace('[', ' ')
        x = x.replace(']', ' ')
        x = x.strip().split()
        ratios.append(float(x[-1]))
    return fname, ratios


csv_data = dict()
with open('labels.csv') as f:
    reader = csv.reader(f)
    for fname, label in reader:
        csv_data[fname] = [int(label)]

with open('image_features4.csv') as f:
    reader = csv.reader(f)
    for row in reader:
        fname, ratios = parse_row(row)
        if fname in csv_data:
            csv_data[fname].append(ratios)

X, y = [], []
for label, ftrs in csv_data.values():
    if label == 1:
        X.append(ftrs)
        y.append(0)
    elif label == 2:
        X.append(ftrs)
        y.append(1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)
clf1 = MLPClassifier(solver='lbfgs', hidden_layer_sizes=(4), max_iter=1000)
clf1.fit(X_train, y_train)
predictions = clf1.predict(X_test)
print(f'{accuracy_score(y_test, predictions) * 100:.2f}')
print(f'{f1_score(y_test, predictions, average=None)}')

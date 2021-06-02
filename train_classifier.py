from sklearn.svm import NuSVC
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score
import csv
import pickle


def parse_row(row):
    fname = row[0]
    centers = []
    for x in row[1:]:
        x = x.replace('[', ' ')
        x = x.replace(']', ' ')
        x = x.strip().split()
        for num in x:
            centers.append(float(num))
    return fname, centers


csv_data = dict()
with open('labels.csv') as f:
    reader = csv.reader(f)
    for fname, label in reader:
        csv_data[fname] = [int(label)]

with open('cluster_centers4.csv') as f:
    reader = csv.reader(f)
    for row in reader:
        fname, centers = parse_row(row)
        if fname in csv_data:
            csv_data[fname].append(centers)

X, y = [], []
for label, ftrs in csv_data.values():
    X.append(ftrs)
    y.append(label)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)
clf = make_pipeline(
    StandardScaler(),
    NuSVC(nu=0.1, class_weight='balanced'))
clf.fit(X_train, y_train)
predictions = clf.predict(X_test)
print(f'{accuracy_score(y_test, predictions) * 100:.2f}')
print(f'{f1_score(y_test, predictions, average=None)}')

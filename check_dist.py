import csv

dist = [0, 0, 0]
with open('labels.csv') as f:
    reader = csv.reader(f)
    for _, label in reader:
        dist[int(label)] += 1

print(dist)

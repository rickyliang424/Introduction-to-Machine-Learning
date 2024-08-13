# -*- coding: utf-8 -*-
""" GaussianNB_scratch """

#%% 1
from csv import reader
from math import sqrt, exp, pi
# from sklearn import datasets
# iris = datasets.load_iris()

path = "C:/Users/Ricky/Desktop/大三下 課程/機器學習導論/TA2/"
filename = path + "iris_dataset.csv"
dataset_1 = []
with open(filename,'r') as file:
    csv_reader = reader(file)
    for row in csv_reader:
        if not row:
            continue
        dataset_1.append(row)

dataset_1

#%% 2
dataset_2 = dataset_1[50:150]
for i in range(len(dataset_2)):
    dataset_2[i].pop(0)
    dataset_2[i].pop(2)

dataset_2

#%% 3
dataset_3 = dataset_2
for i in range(len(dataset_3[0])-1):
    for row in dataset_3:
        row[i] = float(row[i])

column = len(dataset_3[0])-1
class_values = [row[column] for row in dataset_3]
unique = set(class_values)
lookup = dict()
for i, value in enumerate(unique):
    lookup[value] = i
for row in dataset_3:
    row[column] = lookup[row[column]]

lookup
dataset_3

#%% 4
def mean(numbers):
    mean = sum(numbers) / float(len(numbers))
    return mean
def stdev(numbers):
	avg = mean(numbers)
	variance = sum([(x-avg)**2 for x in numbers]) / float(len(numbers)-1)
	return sqrt(variance)
def summarize_dataset(dataset):
	summaries = [(mean(column), stdev(column), len(column)) for column in zip(*dataset)]
	del(summaries[-1])
	return summaries

separated = dict()
for i in range(len(dataset_3)):
	vector = dataset_3[i]
	class_value = vector[-1]
	if (class_value not in separated):
		separated[class_value] = []
	separated[class_value].append(vector)
    
summaries = dict()
for class_value, rows in separated.items():
	summaries[class_value] = summarize_dataset(rows)

separated
summaries

#%% 5
def calculate_probability(x, mean, stdev):
    exponent = exp(-((x-mean)**2 / (2 * stdev**2)))
    pdf = (1 / (sqrt(2*pi) * stdev)) * exponent
    return pdf

row = [3.5,1.4]
total_rows = sum([summaries[label][0][1] for label in summaries])
probabilities = dict()
for class_value, class_summaries in summaries.items():
	probabilities[class_value] = summaries[class_value][0][1] / float(total_rows)
	for i in range(len(class_summaries)):
		MEAN, STDEV, _ = class_summaries[i]
		probabilities[class_value] *= calculate_probability(row[i], MEAN, STDEV)

probabilities

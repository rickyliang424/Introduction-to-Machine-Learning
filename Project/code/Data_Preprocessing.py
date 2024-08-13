# -*- coding: utf-8 -*-
"""
Created on Tue Jun  8 17:41:03 2021
@author: Mey
"""
#data analysis libraries
import numpy as np
import pandas as pd

#visualization libraries
import matplotlib.pyplot as plt
import seaborn as sns

#ignore warnings
import warnings
warnings.filterwarnings('ignore')

#import train and test CSV files
train = pd.read_csv("./train.csv")
test = pd.read_csv("./test.csv")
#take a look at the training data
train.describe(include="all")

#-------------------------------------------------------------------
#Visualize the count of number of survivors
ax=sns.countplot(train['Survived'],label="Count")
for p in ax.patches:
    ax.annotate(f'\n{p.get_height()}', (p.get_x()+0.2, p.get_height()), ha='center', va='top', color='white', size=18)
plt.show()

#%%
#draw a bar plot of survival by sex
sns.barplot(x="Sex", y="Survived", data=train)
#print percentages of females vs. males that survive
print("Percentage of females who survived:", train["Survived"][train["Sex"] == 'female'].value_counts(normalize = True)[1]*100)
print("Percentage of males who survived:", train["Survived"][train["Sex"] == 'male'].value_counts(normalize = True)[1]*100)
#-------------------------------------------------------------------
#map each Sex value to a numerical value
sex_mapping = {"male": 0, "female": 1}
train['Sex'] = train['Sex'].map(sex_mapping)
test['Sex'] = test['Sex'].map(sex_mapping)
#-------------------------------------------------------------------
#draw a bar plot of survival by Pclass
sns.barplot(x="Pclass", y="Survived", data=train)

#print percentage of people by Pclass that survived
print("Percentage of Pclass = 1 who survived:", train["Survived"][train["Pclass"] == 1].value_counts(normalize = True)[1]*100)

print("Percentage of Pclass = 2 who survived:", train["Survived"][train["Pclass"] == 2].value_counts(normalize = True)[1]*100)

print("Percentage of Pclass = 3 who survived:", train["Survived"][train["Pclass"] == 3].value_counts(normalize = True)[1]*100)

data = [train, test]
for dataset in data:
    dataset['Fare'] = dataset['Fare'].fillna(np.mean(dataset['Fare']))
    dataset['relatives'] = dataset['SibSp'] + dataset['Parch']
    dataset['fare_pp'] = dataset['Fare'] / (dataset['relatives'] + 1)

    dataset['Pclass'] = dataset['Pclass'].fillna(0.)
    index1 = np.array(dataset[(dataset['Pclass'] == 0.) & (dataset['fare_pp'] >= 20.)].index)
    index2 = np.array(dataset[(dataset['Pclass'] == 0.) & (dataset['fare_pp'] > 10.) & (dataset['fare_pp'] < 20.)].index)
    index3 = np.array(dataset[(dataset['Pclass'] == 0.) & (dataset['fare_pp'] <= 10.)].index)

    pclass = np.array(dataset['Pclass'])
    for i in range(len(index1)):
        pclass[index1[i]] = 1.
    for i in range(len(index2)):
        pclass[index2[i]] = 2.
    for i in range(len(index3)):
        pclass[index3[i]] = 3.

    dataset['Pclass'] = pd.DataFrame(pclass)

#-------------------------------------------------------------------
#draw a bar plot for SibSp vs. survival
sns.barplot(x="SibSp", y="Survived", data=train)

#I won't be printing individual percent values for all of these.
print("Percentage of SibSp = 0 who survived:", train["Survived"][train["SibSp"] == 0].value_counts(normalize = True)[1]*100)

print("Percentage of SibSp = 1 who survived:", train["Survived"][train["SibSp"] == 1].value_counts(normalize = True)[1]*100)

print("Percentage of SibSp = 2 who survived:", train["Survived"][train["SibSp"] == 2].value_counts(normalize = True)[1]*100)

print("Percentage of SibSp = 3 who survived:", train["Survived"][train["SibSp"] == 3].value_counts(normalize = True)[1]*100)

print("Percentage of SibSp = 4 who survived:", train["Survived"][train["SibSp"] == 4].value_counts(normalize = True)[1]*100)

#-------------------------------------------------------------------
#draw a bar plot for Parch vs. survival
sns.barplot(x="Parch", y="Survived", data=train)
plt.show()

print("Percentage of Parch = 0 who survived:", train["Survived"][train["Parch"] == 0].value_counts(normalize = True)[1]*100)

print("Percentage of Parch = 1 who survived:", train["Survived"][train["Parch"] == 1].value_counts(normalize = True)[1]*100)

print("Percentage of Parch = 2 who survived:", train["Survived"][train["Parch"] == 2].value_counts(normalize = True)[1]*100)

print("Percentage of Parch = 3 who survived:", train["Survived"][train["Parch"] == 3].value_counts(normalize = True)[1]*100)

print("Percentage of Parch = 4 who survived:", train["Survived"][train["Parch"] == 4].value_counts(normalize = True)[1]*100)

#-------------------------------------------------------------------
#label Age and fillin missing value
data = [train,test]
for dataset in data:
    mean = dataset["Age"].mean()
    std = dataset["Age"].std()
    is_null = dataset["Age"].isnull().sum()
    # compute random numbers between the mean, std and is_null
    rand_age = np.random.randint(mean - std, mean + std, size = is_null)
    # fill NaN values in Age column with random values generated
    age_slice = dataset["Age"].copy()
    age_slice[np.isnan(age_slice)] = rand_age
    dataset["Age"] = age_slice
    dataset["Age"] = train["Age"].astype(int)

bins = [-1, 5, 12, 18, 24, 35, 60, np.inf]
labels = ['Baby', 'Child', 'Teenager', 'Student', 'Young Adult', 'Adult', 'Senior']
train['AgeGroup'] = pd.cut(train["Age"], bins, labels = labels)
test['AgeGroup'] = pd.cut(test["Age"], bins, labels = labels)

#map each Age value to a numerical value
age_mapping = {'Baby': 1, 'Child': 2, 'Teenager': 3, 'Student': 4, 'Young Adult': 5, 'Adult': 6, 'Senior': 7}
train['AgeGroup'] = train['AgeGroup'].map(age_mapping)
test['AgeGroup'] = test['AgeGroup'].map(age_mapping)

#draw a bar plot of survival by AgeGroup
sns.barplot(x="AgeGroup", y="Survived", data=train)

#print percentage of people by FareBand that survived
print("Percentage of AgeGroup = 1 who survived:", train["Survived"][train["AgeGroup"] == 1].value_counts(normalize = True)[1]*100)

print("Percentage of AgeGroup = 2 who survived:", train["Survived"][train["AgeGroup"] == 2].value_counts(normalize = True)[1]*100)

print("Percentage of AgeGroup = 3 who survived:", train["Survived"][train["AgeGroup"] == 3].value_counts(normalize = True)[1]*100)

print("Percentage of AgeGroup = 4 who survived:", train["Survived"][train["AgeGroup"] == 4].value_counts(normalize = True)[1]*100)

print("Percentage of AgeGroup = 5 who survived:", train["Survived"][train["AgeGroup"] == 5].value_counts(normalize = True)[1]*100)

print("Percentage of AgeGroup = 6 who survived:", train["Survived"][train["AgeGroup"] == 6].value_counts(normalize = True)[1]*100)

print("Percentage of AgeGroup = 7 who survived:", train["Survived"][train["AgeGroup"] == 7].value_counts(normalize = True)[1]*100)

#-------------------------------------------------------------------
#now we need to fill in the missing values in the Embarked feature
print("Number of people embarking in Southampton (S):")
southampton = train[train["Embarked"] == "S"].shape[0]
print(southampton)

print("Number of people embarking in Cherbourg (C):")
cherbourg = train[train["Embarked"] == "C"].shape[0]
print(cherbourg)

print("Number of people embarking in Queenstown (Q):")
queenstown = train[train["Embarked"] == "Q"].shape[0]
print(queenstown)

#replacing the missing values in the Embarked feature with S
#It's clear that the majority of people embarked in Southampton (S). Let's go ahead and fill in the missing values with S.
test = test.fillna({"Embarked": "S"})
#map each Embarked value to a numerical value
embarked_mapping = {"S": 1, "C": 2, "Q": 3}
train['Embarked'] = train['Embarked'].map(embarked_mapping)
test['Embarked'] = test['Embarked'].map(embarked_mapping)
#draw a bar plot of survival by Embarked
sns.barplot(x="Embarked", y="Survived", data=train)

#print percentage of people by Embarked that survived
print("Percentage of Embarked = 1 who survived:", train["Survived"][train["Embarked"] == 1].value_counts(normalize = True)[1]*100)

print("Percentage of Embarked = 2 who survived:", train["Survived"][train["Embarked"] == 2].value_counts(normalize = True)[1]*100)

print("Percentage of Embarked = 3 who survived:", train["Survived"][train["Embarked"] == 3].value_counts(normalize = True)[1]*100)

#-------------------------------------------------------------------
# label the Name according to the first letter of last name
# Train dataset
name_letter = train['Name'].map(lambda x: ord(x[0].upper())-64)
name_letter = name_letter.rename('Name_Letter')
train=pd.concat([train, name_letter], axis=1)
# label the Name according to the first letter of last name
# Test dataset
name_letter = test['Name'].map(lambda x: ord(x[0].upper())-64)
name_letter = name_letter.rename('Name_Letter')
test=pd.concat([test, name_letter], axis=1)

#-------------------------------------------------------------------
#fill in missing Fare value in mean fare

for x in range(len(train["Fare"])):
    if pd.isnull(train["Fare"][x]):
        train["Fare"][x] = round(train["Fare"].mean(), 4)
#-------------------------------------------------------------------
#Creat a new feature: # of relatives and Along or not alone
data = [train, test]
for dataset in data:
    dataset['relatives'] = dataset['SibSp'] + dataset['Parch']
    dataset.loc[dataset['relatives'] > 0, 'not_alone'] = 0
    dataset.loc[dataset['relatives'] == 0, 'not_alone'] = 1
    dataset['not_alone'] = dataset['not_alone'].astype(int)
train['not_alone'].value_counts()

print(pd.isnull(train).sum())
print(pd.isnull(test).sum())

#-------------------------------------------------------------------
#Cleaning Data
train = train.drop(['PassengerId'], axis = 1)
test = test.drop(['PassengerId'], axis = 1)
train = train.drop(['Cabin'], axis = 1)
test = test.drop(['Cabin'], axis = 1)
train = train.drop(['Ticket'], axis = 1)
test = test.drop(['Ticket'], axis = 1)
train = train.drop(['Name'], axis = 1)
test = test.drop(['Name'], axis = 1)
train = train.drop(['Age'], axis = 1)
test = test.drop(['Age'], axis = 1)
train = train.drop(['Fare'], axis = 1)
test = test.drop(['Fare'], axis = 1)

train.to_csv ('train_Team12.csv', index = False, header=True)
test.to_csv ('test_Team12.csv', index = False, header=True)


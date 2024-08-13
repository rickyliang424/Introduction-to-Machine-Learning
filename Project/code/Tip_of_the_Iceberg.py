# -*- coding: utf-8 -*-
"""
Created on Mon May 17 13:14:33 2021
@author: Ricky
"""
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Setting up visualisations
sns.set_style(style='white') 
sns.set(rc={'figure.figsize':(12,7), 'axes.facecolor':'white', 'axes.grid':True, 
            'grid.color':'.9', 'axes.linewidth':1.0, 'grid.linestyle':u'-'},font_scale=1.5)
custom_colors = ["#3498db", "#95a5a6","#34495e", "#2ecc71", "#e74c3c"]
sns.set_palette(custom_colors)

csv = pd.read_csv('./train3.csv')
category = csv.nunique()

plt.figure()
null_sum = csv.isnull().sum()
pd.DataFrame(null_sum).plot.line().set_title("Number of missing values in the given features")

#%% Survived
plt.figure()
(csv.Survived.value_counts(normalize=True)*100).plot.barh().set_title("Percentage of people survived and Deceased")

#%% Pclass
plt.figure()
fig_pclass = csv.Pclass.value_counts().plot.pie().legend(labels=["Class 3","Class 1","Class 2"])

plt.figure()
pclass_1_survivor_distribution = round((csv[csv.Pclass == 1].Survived == 1).value_counts()[1]/len(csv[csv.Pclass == 1]) * 100, 2)
pclass_2_survivor_distribution = round((csv[csv.Pclass == 2].Survived == 1).value_counts()[1]/len(csv[csv.Pclass == 2]) * 100, 2)
pclass_3_survivor_distribution = round((csv[csv.Pclass == 3].Survived == 1).value_counts()[1]/len(csv[csv.Pclass == 3]) * 100, 2)
pclass_perc_df = pd.DataFrame(
    { "Percentage Survived":{"Class 1": pclass_1_survivor_distribution,"Class 2": pclass_2_survivor_distribution, "Class 3": pclass_3_survivor_distribution},  
     "Percentage Not Survived":{"Class 1": 100-pclass_1_survivor_distribution,"Class 2": 100-pclass_2_survivor_distribution, "Class 3": 100-pclass_3_survivor_distribution}})
pclass_perc_df.plot.bar().set_title("Percentage of people survived on the basis of class")

plt.figure()
for x in [1,2,3]:
    csv.Age[csv.Pclass == x].plot(kind="kde")
plt.title("Age density in classes")
plt.legend(("1st","2nd","3rd"))

plt.figure()
for x in [1,0]:
    csv.Pclass[csv.Sex == x].plot(kind="kde")
plt.title("Gender density in classes")
plt.legend(("Male","Female"))

#%% Sex
plt.figure()
(csv.Sex.value_counts(normalize = True) * 100).plot.bar().set_title("Number of male and female")
plt.xlabel("1.0 => male  ,  0.0 => female")

plt.figure()
male_pr = round((csv[csv.Sex == 1].Survived == 1).value_counts()[1]/len(csv.Sex) * 100, 2)
female_pr = round((csv[csv.Sex == 0].Survived == 1).value_counts()[1]/len(csv.Sex) * 100, 2)
sex_perc_df = pd.DataFrame(
    { "Percentage Survived":{"male": male_pr,"female": female_pr}, "Percentage Not Survived":{"male": 100-male_pr,"female": 100-female_pr}})
sex_perc_df.plot.barh().set_title("Percentage of male and female survived and Deceased")

#%% Age
age_info = pd.DataFrame(csv.Age.describe())

plt.figure()
csv['Age_Range'] = pd.cut(csv.Age, [0, 10, 20, 30, 40, 50, 60,70,80])
sns.countplot(x = "Age_Range", hue = "Survived", data = csv, palette=["C1", "C0"]).legend(labels = ["Deceased", "Survived"])

plt.figure()
sns.distplot(csv['Age'].dropna(),color='darkgreen',bins=30)

#%% SibSp
sibsp_info = csv.SibSp.describe()

plt.figure()
ss = pd.DataFrame()
ss['survived'] = csv.Survived
ss['sibling_spouse'] = pd.cut(csv.SibSp, [0, 1, 2, 3, 4, 5, 6,7,8], include_lowest = True)
(ss.sibling_spouse.value_counts()).plot.area().set_title("Number of siblings or spouses vs survival count")

plt.figure()
x = sns.countplot(x = "sibling_spouse", hue = "survived", data = ss, palette=["C1", "C0"]).legend(labels = ["Deceased", "Survived"])
x.set_title("Survival based on number of siblings or spouses")

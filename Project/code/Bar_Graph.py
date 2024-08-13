# -*- coding: utf-8 -*-
"""
Created on Sat May 15 16:45:24 2021
@author: Ricky
"""
# 參考資料：https://towardsdatascience.com/a-beginners-guide-to-kaggle-s-titanic-problem-3193cb56f6ca
# 參考資料：https://github.com/sumitmukhija/Titanic
# 參考資料：Tip of the Iceberg.py , Tip of the Iceberg.ipynb
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Setting up visualisations
sns.set_style(style='white') 
sns.set(rc={'figure.figsize':(12,7), 'axes.facecolor':'white', 'axes.grid':True, 
            'grid.color':'.9', 'axes.linewidth':1.0, 'grid.linestyle':u'-'},font_scale=1.5)
custom_colors = ["#3498db", "#95a5a6","#34495e", "#2ecc71", "#e74c3c"]
sns.set_palette(custom_colors)

train4 = pd.read_csv('./train4.csv').iloc[0:891]
category = train4.nunique()

#%% Pclass
plt.figure()
pclass_1_survivor_distribution = round((train4[train4.Pclass == 1].Survived == 1).value_counts()[1]/len(train4[train4.Pclass == 1]) * 100, 2)
pclass_2_survivor_distribution = round((train4[train4.Pclass == 2].Survived == 1).value_counts()[1]/len(train4[train4.Pclass == 2]) * 100, 2)
pclass_3_survivor_distribution = round((train4[train4.Pclass == 3].Survived == 1).value_counts()[1]/len(train4[train4.Pclass == 3]) * 100, 2)
pclass_perc_df = pd.DataFrame(
    { "Percentage Survived":{"Class 1": pclass_1_survivor_distribution, "Class 2": pclass_2_survivor_distribution, "Class 3": pclass_3_survivor_distribution},  
     "Percentage Not Survived":{"Class 1": 100-pclass_1_survivor_distribution, "Class 2": 100-pclass_2_survivor_distribution, "Class 3": 100-pclass_3_survivor_distribution}})
pclass_perc_df.plot.bar().set_title("Percentage of people survived on the basis of class")

#%% Name
plt.figure()
name_survivor_distribution = np.zeros(27)
for i in range(1,27,1):
    if len((train4[train4.Name == i].Survived == 1).value_counts()) == 2:
        name_survivor_distribution[i] = round((train4[train4.Name == i].Survived == 1).value_counts()[1]/len(train4[train4.Name == 1]) * 100, 2)
name_perc_df = pd.DataFrame(
    { "Percentage Survived":{"A~": name_survivor_distribution[1], "B~": name_survivor_distribution[2], 
                             "C~": name_survivor_distribution[3], "D~": name_survivor_distribution[4], 
                             "E~": name_survivor_distribution[5], "F~": name_survivor_distribution[6], 
                             "G~": name_survivor_distribution[7], "H~": name_survivor_distribution[8], 
                             "I~": name_survivor_distribution[9], "J~": name_survivor_distribution[10], 
                             "K~": name_survivor_distribution[11], "L~": name_survivor_distribution[12], 
                             "M~": name_survivor_distribution[13], "N~": name_survivor_distribution[14], 
                             "O~": name_survivor_distribution[15], "P~": name_survivor_distribution[16], 
                             "Q~": name_survivor_distribution[17], "R~": name_survivor_distribution[18], 
                             "S~": name_survivor_distribution[19], "T~": name_survivor_distribution[20], 
                             "U~": name_survivor_distribution[21], "V~": name_survivor_distribution[22], 
                             "W~": name_survivor_distribution[23], "X~": name_survivor_distribution[24], 
                             "Y~": name_survivor_distribution[25], "Z~": name_survivor_distribution[26],},  
     "Percentage Not Survived":{"A~": 100-name_survivor_distribution[1], "B~": 100-name_survivor_distribution[2], 
                                "C~": 100-name_survivor_distribution[3], "D~": 100-name_survivor_distribution[4], 
                                "E~": 100-name_survivor_distribution[5], "F~": 100-name_survivor_distribution[6], 
                                "G~": 100-name_survivor_distribution[7], "H~": 100-name_survivor_distribution[8], 
                                "I~": 100-name_survivor_distribution[9], "J~": 100-name_survivor_distribution[10], 
                                "K~": 100-name_survivor_distribution[11], "L~": 100-name_survivor_distribution[12], 
                                "M~": 100-name_survivor_distribution[13], "N~": 100-name_survivor_distribution[14], 
                                "O~": 100-name_survivor_distribution[15], "P~": 100-name_survivor_distribution[16], 
                                "Q~": 100-name_survivor_distribution[17], "R~": 100-name_survivor_distribution[18], 
                                "S~": 100-name_survivor_distribution[19], "T~": 100-name_survivor_distribution[20], 
                                "U~": 100-name_survivor_distribution[21], "V~": 100-name_survivor_distribution[22], 
                                "W~": 100-name_survivor_distribution[23], "X~": 100-name_survivor_distribution[24], 
                                "Y~": 100-name_survivor_distribution[25], "Z~": 100-name_survivor_distribution[26],}})
name_perc_df.plot.bar().set_title("Percentage of people survived on the basis of name")

#%% Sex
plt.figure()
male_pr = round((train4[train4.Sex == 1].Survived == 1).value_counts()[1]/len(train4.Sex) * 100, 2)
female_pr = round((train4[train4.Sex == 0].Survived == 1).value_counts()[1]/len(train4.Sex) * 100, 2)
sex_perc_df = pd.DataFrame(
    { "Percentage Survived":{"male": male_pr,"female": female_pr}, "Percentage Not Survived":{"male": 100-male_pr,"female": 100-female_pr}})
sex_perc_df.plot.barh().set_title("Percentage of male and female survived and Deceased")

#%% Age
plt.figure()
train4['Age_Range'] = pd.cut(train4.Age, [0, 10, 20, 30, 40, 50, 60, 70, 80])
sns.countplot(x="Age_Range", hue="Survived", data=train4, palette=["C1","C0"]).legend(labels=["Deceased","Survived"])
plt.title("Survival based on interval of age")

#%% SibSp
plt.figure()
ss = pd.DataFrame()
ss['survived'] = train4.Survived
ss['sibling_spouse'] = pd.cut(train4.SibSp, [0, 1, 2, 3, 4, 5, 6,7,8], include_lowest = True)
sns.countplot(x="sibling_spouse", hue="survived", data=ss, palette=["C1","C0"]).legend(labels=["Deceased","Survived"])
plt.title("Survival based on number of siblings or spouses")

#%% Fare
fare_info = train4.Fare.describe()
train4['Fare_Category'] = pd.cut(train4['Fare'], bins=[0,7.90,14.45,31.28,120], labels=['Low','Mid','High_Mid','High'])
sns.countplot(x="Fare_Category", hue="Survived", data=train4, palette=["C1","C0"]).legend(labels=["Deceased","Survived"])
plt.title("Survival based on fare category")

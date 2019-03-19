"""
Toxic Comment EDA

"""
# ---- IMPORT PACKAGES ------ #
import pandas as pd
import numpy as np

# data viz
import matplotlib.pyplot as plt
import seaborn as sns

# import data
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

# -- INITIAL LOOK AT DATA -- #
# Look at data
train.head()
train.describe()
train.dtypes

# Check missing values in train/test datasets
print("Number of missing value in train data:")
print(train.isnull().sum())

print("Number of missing values in test data:")
print(test.isnull().sum())
print("filling NA with \"unknown\"")

# LN: No missing data to deal with.

# Look at the distribution of toxic comments
x = train.iloc[:, 2:].sum()

# marking comments without any tags as "clean"
rowsums = train.iloc[:, 2:].sum(axis=1)
train['not-toxic'] = (rowsums == 0)
# count number of clean entries
train['not-toxic'].sum()

print("Total comments = ", len(train))
print("Total clean comments = ", train['not-toxic'].sum())
print("Total tags =", x.sum())

# plot
x = train.iloc[:, 2:].sum()

plt.figure(figsize=(8, 4))
ax = sns.barplot(x.index, x.values, alpha=0.8)
plt.title("# per class")
plt.ylabel('# of Occurrences', fontsize=12)
plt.xlabel('Type ', fontsize=12)
# adding the text labels
rects = ax.patches
labels = x.values
# adding value labels to the bars
for rect, label in zip(rects, labels):
    height = rect.get_height()
    ax.text(rect.get_x() + rect.get_width()/2, height + 5, label,
            ha='center', va='bottom')

plt.show()
# 1- There are significantly more non-toxic comments than toxic ones combined.
# 2- Toxicity is not distributed evenly across toxicity classes

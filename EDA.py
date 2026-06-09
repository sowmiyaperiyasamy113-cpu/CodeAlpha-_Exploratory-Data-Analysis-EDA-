import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
df = pd.read_csv("dataset.csv")
df.head()
df.tail()
df.info()
df.describe()
df.isnull().sum()
sns.heatmap(df.isnull(), cbar=False)
plt.show()
df.duplicated().sum()
df.drop_duplicates(inplace=True)
df.dtypes
numerical_cols = df.select_dtypes(include=np.number).columns

categorical_cols = df.select_dtypes(include='object').columns
df.hist(figsize=(12,8))
plt.show()
for col in numerical_cols:
    plt.figure(figsize=(5,3))
    sns.boxplot(x=df[col])
    plt.title(col)
    plt.show()
    for col in categorical_cols:
    sns.countplot(x=df[col])
    plt.xticks(rotation=45)
    plt.show()
    corr = df.corr(numeric_only=True)
corr
plt.figure(figsize=(10,8))
sns.heatmap(corr,
            annot=True,
            cmap='coolwarm')
plt.show()
sns.scatterplot(x='Feature1',
                y='Feature2',
                data=df)
plt.show()
sns.pairplot(df)
plt.show()
Q1 = df['column'].quantile(0.25)
Q3 = df['column'].quantile(0.75)

IQR = Q3 - Q1

lower = Q1 - 1.5 * IQR
upper = Q3 + 1.5 * IQR

outliers = df[(df['column'] < lower) |
              (df['column'] > upper)]

print(outliers)
from scipy.stats import pearsonr

corr,p = pearsonr(df['FeatureA'],
                  df['Target'])

print("Correlation:",corr)
print("P-value:",p)
if p < 0.05:
    print("Reject H0")
else:
    print("Accept H0")
    sns.lineplot(x='Date',
             y='Sales',
             data=df)

plt.show()
df.isnull().sum()
df.duplicated().sum()
sns.boxplot(data=df)
plt.show()
df.dtypes
    

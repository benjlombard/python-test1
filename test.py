#!/usr/bin/env/python3

"""
first programm

for installing pandas in pycharm
sudo apt-get install python-dev   # for python2.x installs
sudo apt-get install python3-dev  # for python3.x installs

For yum (CentOS, RHEL...):

sudo yum install python-devel   # for python2.x installs
sudo yum install python34-devel   # for python3.4 installs

For dnf (Fedora...):

sudo dnf install python2-devel  # for python2.x installs
sudo dnf install python3-devel  # for python3.x installs

For zypper (openSUSE...):

sudo zypper in python-devel   # for python2.x installs
sudo zypper in python3-devel  # for python3.x installs

For apk (Alpine...):

# This is a departure from the normal Alpine naming
# scheme, which uses py2- and py3- prefixes
sudo apk add python2-dev  # for python2.x installs
sudo apk add python3-dev  # for python3.x installs

"""
#le module seaborn est pour la visualisation statistique, seaborn est basé sur matplotlib
#install tkinter : sudo yum -y install tkinter tcl-devel tk-devel
#sudo yum install -y python34-tkinter => pour pouvoir tracer des graphiques avec matplotlib

import sys
import os
import numpy as np
import pandas as pd
import requests
import matplotlib.pyplot as plt
import seaborn as sns
import patsy
import statsmodels.api as sm
from sklearn.svm import LinearSVC

r = requests.get("https://api.github.com/users/gcaggia/starred")
print(r.json())

r = requests.get("https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data")
PATH = r'/home/ben/iris/'
with open(PATH + 'iris.data', 'w') as f:
    f.write(r.text)
os.chdir(PATH)
df = pd.read_csv(PATH + 'iris.data', names = ['sepal length', 'sepal width', 'petal length', 'petal width', 'class'])

print(df.head())
print(df.tail())
print(df.count())
print(df['class'].unique())
print(df[df['class'] == 'Iris-setosa'].count())
print(df[df['class'] == 'Iris-setosa'])
print(df[df['class'] == 'Iris-virginica'].reset_index(drop=True))
print(df[df['class'] == 'Iris-virginica'])
print(df.ix[:3, :2])
print(df.ix[:3, [x for x in df.columns if 'width' in x]])
print(df['sepal length'])
print(df[(df['class'] == 'Iris-virginica') & (df['petal length'] > 6.2)].reset_index(drop=True))
print(df[(df['class'] == 'Iris-virginica') & (df['petal length'] > 6.2)])
#par défaut method = pearson, on peut choisir aussi spearman ou kendall
#print(df.corr(method="kendall"))
print(df.corr(method="spearman"))
print(df.corr(method="pearson"))
print(df.describe())
print(df.describe(percentiles=[.25, .4, .8, .9, .95]))
plt.style.use('ggplot')
fig, ax = plt.subplots(figsize=(6,4))
ax.hist(df['petal width'], color='black')
ax.set_ylabel('Count', fontsize=12)
ax.set_xlabel('Width', fontsize=12)
plt.title('Iris Petal Width', fontsize=14, y=1.01)
plt.show()


fig1, ax = plt.subplots(2,2,figsize=(6,4))
ax[0][0].hist(df['petal width'], color='black')
ax[0][0].set_ylabel('Count', fontsize=12)
ax[0][0].set_xlabel('Width', fontsize=12)
ax[0][0].set_title('Iris Petal Width', fontsize=14, y=1.01)

ax[0][1].hist(df['petal length'], color='black')
ax[0][1].set_ylabel('Count', fontsize=12)
ax[0][1].set_xlabel('Length', fontsize=12)
ax[0][1].set_title('Iris Petal Length', fontsize=14, y=1.01)

ax[1][0].hist(df['sepal width'], color='black')
ax[1][0].set_ylabel('Count', fontsize=12)
ax[1][0].set_xlabel('Wdith', fontsize=12)
ax[1][0].set_title('Iris Sepal Width', fontsize=14, y=1.01)

ax[1][1].hist(df['sepal length'], color='black')
ax[1][1].set_ylabel('Count', fontsize=12)
ax[1][1].set_xlabel('Length', fontsize=12)
ax[1][1].set_title('Iris Sepal Length', fontsize=14, y=1.01)


fig2, ax = plt.subplots(figsize=(6, 6))
ax.scatter(df['petal width'], df['petal length'], color='green')
ax.set_xlabel('Petal Width')
ax.set_ylabel('Petal Length')
ax.set_title('Petal Scatterplot')


fig3, ax = plt.subplots(figsize=(6,6))
ax.plot(df['petal length'], color='blue')
ax.set_xlabel('Specimen Number')
ax.set_ylabel('Petal Length')
ax.set_title('Petal Length Plot')



sns.pairplot(df,hue='class')

plt.tight_layout()
plt.show()



print("hello world")






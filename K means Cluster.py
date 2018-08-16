#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  6 12:59:10 2018

@author: shilinli
"""

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df_college = pd.read_csv('College_Data',index_col=0)

# Grad.Rate versus Room.Board where the points are colored by the Private column
sns.set_style('whitegrid')
sns.lmplot('Room.Board','Grad.Rate',data=df_college, hue='Private',
           palette='coolwarm',size=6,aspect=1,fit_reg=False)

# Scatterplot of F.Undergrad versus Outstate where the points are colored by the Private column
sns.set_style('whitegrid')
sns.lmplot('Outstate','F.Undergrad',data=df_college, hue='Private',
           palette='coolwarm',size=6,aspect=1,fit_reg=False)

# Histogram showing Out of State Tuition based on the Private column
sns.set_style('darkgrid')
g = sns.FacetGrid(df_college,hue="Private",palette='coolwarm',size=6,aspect=2)
g = g.map(plt.hist,'Outstate',bins=20,alpha=0.7)

# Similar histogram for the Grad.Rate column
sns.set_style('darkgrid')
g = sns.FacetGrid(df_college,hue="Private",palette='coolwarm',size=6,aspect=2)
g = g.map(plt.hist,'Grad.Rate',bins=20,alpha=0.7)

# Check which university has graduation rate greater than 100
df_college[df_college['Grad.Rate'] > 100]

# Set its graduation rate to 100
df_college['Grad.Rate']['Cazenovia College'] = 100

# K Means Clustering
from sklearn.cluster import KMeans
k_means = KMeans(n_clusters=2)
k_means.fit(df_college.drop('Private',axis = 1))

# Center of K-means
k_means.cluster_centers_
k_means.labels_

# Evaluations

# Create a cluster column showing 1 to private and 0 to non-private
def converter(cluster):
    if cluster=='Yes':
        return 1
    else:
        return 0

df_college['Cluster'] = df_college['Private'].apply(converter)

from sklearn.metrics import classification_report,confusion_matrix
confusion_matrix(df_college['Cluster'],k_means.labels_)
classification_report(df_college['Cluster'],k_means.labels_)

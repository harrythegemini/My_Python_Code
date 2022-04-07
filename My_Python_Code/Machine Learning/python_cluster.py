#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 11 17:17:35 2021

@author: haoyuwang
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import KMeans
from sklearn import datasets
from sklearn import preprocessing
import pylab as pl
from sklearn import decomposition
import seaborn as sns
import sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn import metrics
from sklearn.cluster import AgglomerativeClustering

import scipy.cluster.hierarchy as hc
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer

## Vectorization
# Read in textdata as dataframe
path="/Users/haoyuwang/Downloads/harrywang/GU/Courses/501/Cluster/"
filelocation="textdata copy.csv"
df=pd.read_csv(path+filelocation)
print(df)

# CountVectorizer
CV_headline=CountVectorizer(input='c',
                        stop_words='english',
                        #max_features=100
                        )
DTM=CV_headline.fit_transform(df['Headline'])
print(DTM)
ColNames=CV_headline.get_feature_names()
print("The vocab is: ", ColNames, "\n\n")

# Tfidf Vectorizer
Tf_headline= TfidfVectorizer(stop_words="english")
DTM2 = Tf_headline.fit_transform(df['Headline'])
print(DTM2)


### Find the optimal K value
# Read data
path="/Users/haoyuwang/Downloads/harrywang/GU/Courses/501/Data Gathering/"
filename = "clean_text_data.csv"
df2= pd.read_csv(path+filename)
df2
# Remove labels
df2= df2.drop(['News Source'],axis=1)
print(df2)

# Elbow
SS_dist = []

values_for_k=range(2,7)
print(values_for_k)

for k_val in values_for_k:
    print(k_val)
    k_means = KMeans(n_clusters=k_val)
    model = k_means.fit(df2)
    SS_dist.append(k_means.inertia_)
    
print(SS_dist)
print(values_for_k)

plt.plot(values_for_k, SS_dist, 'bx-')
plt.xlabel('value')
plt.ylabel('Sum of squared distances')
plt.title('Elbow method for optimal k Choice')
plt.savefig('Elbow.png')
plt.show()


# Silhouette
Sih=[]
Cal=[]
k_range=range(2,8)

for k in k_range:
    kmeans_n=KMeans(n_clusters=k)
    model = kmeans_n.fit(df2)
    prediction = kmeans_n.predict(df2)
    labels = kmeans_n.labels_
    R1 = metrics.silhouette_score(df2,labels,metric='euclidean')
    R2 = metrics.calinski_harabasz_score(df2,labels)
    Sih.append(R1)
    Cal.append(R2)
    
print(Sih)
print(Cal)

fig1,(ax1) = plt.subplots(nrows=1,ncols=1)
ax1.plot(k_range,Sih)
ax1.set_title("Silhouette")
ax1.set_xlabel("")
fig1,(ax2) = plt.subplots(nrows=1,ncols=1)
ax2.plot(k_range,Cal)
ax2.set_title("Calinski_Harabasz_Score")
ax2.set_xlabel("k values")



### Clustering 
## K-Means
############### K=3 ################
kmeans = KMeans(n_clusters=3,init='k-means++',max_iter=200,n_init=1)
kmeans.fit(DTM)
labels = kmeans.labels_
print(labels)

centroids = kmeans.cluster_centers_
print(centroids)

# Normalization
df2_normalized=(df2 - df2.mean()) / df2.std()
print(df2_normalized)

## It seems hard to visualize. Leveraged PCA
print(df2.shape[0])   ## num rows
print(df2.shape[1])   ## num cols

NumCols=df2.shape[1]

# Instantiated my own copy of PCA
My_pca = PCA(n_components=2)  ## I want the two prin columns

# Transpose it
df2_normalized=np.transpose(df2_normalized)
My_pca.fit(df2_normalized)

print(My_pca)
print(My_pca.components_.T)

# Reformat and view results

Comps = pd.DataFrame(My_pca.components_.T,
                        columns=['PC%s' % _ for _ in range(2)],
                        index=df2_normalized.columns
                        )
print(Comps)
print(Comps.iloc[:,0])
RowNames = list(Comps.index)
print(RowNames)

# See 2D PCA Clusters plt.figure(figsize=(12,12))

plt.scatter(Comps.iloc[:,0], Comps.iloc[:,1], s=1, c=labels.astype(float))

plt.xlabel("PC 1")
plt.ylabel("PC 2")
plt.title("Scatter Plot Clusters PC 1 and 2",fontsize=15)

plt.savefig('txt_PCA_k3.png')
plt.show()

############### K=4 ################
kmeans = KMeans(n_clusters=4,init='k-means++',max_iter=200,n_init=1)
kmeans.fit(DTM)
labels = kmeans.labels_
print(labels)

centroids = kmeans.cluster_centers_
print(centroids)

# Normalization
df2_normalized=(df2 - df2.mean()) / df2.std()
print(df2_normalized)

## It seems hard to visualize. Leveraged PCA
print(df2.shape[0])   ## num rows
print(df2.shape[1])   ## num cols

NumCols=df2.shape[1]

# Instantiated my own copy of PCA
My_pca = PCA(n_components=2)  ## I want the two prin columns

# Transpose it
df2_normalized=np.transpose(df2_normalized)
My_pca.fit(df2_normalized)

print(My_pca)
print(My_pca.components_.T)

# Reformat and view results

Comps = pd.DataFrame(My_pca.components_.T,
                        columns=['PC%s' % _ for _ in range(2)],
                        index=df2_normalized.columns
                        )
print(Comps)
print(Comps.iloc[:,0])
RowNames = list(Comps.index)
print(RowNames)

# See 2D PCA Clusters plt.figure(figsize=(12,12))

plt.scatter(Comps.iloc[:,0], Comps.iloc[:,1], s=1, c=labels.astype(float))

plt.xlabel("PC 1")
plt.ylabel("PC 2")
plt.title("Scatter Plot Clusters PC 1 and 2",fontsize=15)

plt.savefig('txt_PCA_k4.png')
plt.show()


############### K=5 ################
kmeans = KMeans(n_clusters=5,init='k-means++',max_iter=200,n_init=1)
kmeans.fit(DTM)
labels = kmeans.labels_
print(labels)

centroids = kmeans.cluster_centers_
print(centroids)

# Normalization
df2_normalized=(df2 - df2.mean()) / df2.std()
print(df2_normalized)

## It seems hard to visualize. Leveraged PCA
print(df2.shape[0])   ## num rows
print(df2.shape[1])   ## num cols

NumCols=df2.shape[1]

# Instantiated my own copy of PCA
My_pca = PCA(n_components=2)  ## I want the two prin columns

# Transpose it
df2_normalized=np.transpose(df2_normalized)
My_pca.fit(df2_normalized)

print(My_pca)
print(My_pca.components_.T)

# Reformat and view results

Comps = pd.DataFrame(My_pca.components_.T,
                        columns=['PC%s' % _ for _ in range(2)],
                        index=df2_normalized.columns
                        )
print(Comps)
print(Comps.iloc[:,0])
RowNames = list(Comps.index)
print(RowNames)

# See 2D PCA Clusters plt.figure(figsize=(12,12))

plt.scatter(Comps.iloc[:,0], Comps.iloc[:,1], s=1, c=labels.astype(float))

plt.xlabel("PC 1")
plt.ylabel("PC 2")
plt.title("Scatter Plot Clusters PC 1 and 2",fontsize=15)

plt.savefig('txt_PCA_k5.png')
plt.show()



## Hierarchical clustering
MyHC = AgglomerativeClustering(n_clusters=4, affinity='euclidean', linkage='ward')
FIT=MyHC.fit(df2)
HC_labels = MyHC.labels_
print(HC_labels)

plt.figure(figsize =(12, 12))
plt.title('Hierarchical Clustering')
dendro = hc.dendrogram((hc.linkage(df2, method ='ward')))
plt.savefig('hierarchy.png')
from sklearn.metrics.pairwise import euclidean_distances
EDist=euclidean_distances(df2)
print(EDist)

## DBSCAN
from sklearn.cluster import DBSCAN
from sklearn import metrics
MyDBSCAN = DBSCAN(eps=6, min_samples=2)
MyDBSCAN.fit_predict(df2)
print(MyDBSCAN.labels_)

# Reduce dimension by PCA
DB_pca = PCA(n_components=2)  ## I want the two prin columns

# Transpose it
df2_normalized=np.transpose(df2_normalized)
DB_pca.fit(df2_normalized)
pca_2d = pca.transform(df2)

print(DB_pca)
print(DB_pca.components_.T)

# Reformat and view results

Comps_db = pd.DataFrame(DB_pca.components_.T,
                        columns=['PC%s' % _ for _ in range(2)],
                        index=df2_normalized.columns
                        )
print(Comps_db)
print(Comps_db.iloc[:,0])
RowNames = list(Comps_db.index)
print(RowNames)

#plot
plt.scatter(pca_2d[:,0], pca_2d[:,1], c=MyDBSCAN.labels_, s=100)
plt.xlabel("PC 1")
plt.ylabel("PC 2")
plt.title("Scatter Plot DBSCAN Clusters PC 1 and 2",fontsize=15)
plt.savefig('DBSACN.png')
plt.show()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  4 13:15:54 2021

@author: haoyuwang
"""

#2.Make content into df
import pandas as pd
import os
import glob
df=pd.DataFrame()
for file in glob.iglob(r'/Users/haoyuwang/Downloads/harrywang/GU/Courses/501/Disscusion_3/*.txt'):
    term = file.split("/")[-1].replace(".txt", "")
    file = open(file)
    content = file.read()
    file.close()
    df = df.append([[term, content]])
df.columns = ["term","content"]


#3.Vectorize
from sklearn.feature_extraction.text import TfidfVectorizer
vector = TfidfVectorizer(stop_words="english")
x = vector.fit_transform(df['content'])
print(x)


#4. Apply clustering model
from sklearn.cluster import KMeans
model = KMeans(n_clusters=3,init='k-means++',max_iter=200,n_init=1)
model.fit(x)
labels = model.labels_
df['label'] = labels


#5.plotting Wordcloud
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import PIL.Image as image
import numpy as np

df0 = df[df.label==0]
content0 = df0['content']
print(content0)
content0list = '\t'.join([word for word in content0])
print(content0list)
mask = np.array(image.open("bitcoin.png"))
wd0 = WordCloud(mask=mask,max_font_size=50, max_words=1000, background_color="white").generate(content0list)
plt.title("Cluster: cryptocurrecny")
plt.axis("off")
plt.imshow(wd0)
plt.show()


df1 = df[df.label==1]
content1 = df1['content']
content1list = '\t'.join([word for word in content1])
mask = np.array(image.open("centralbank.jpeg"))
wd1 = WordCloud(mask=mask,max_font_size=50, max_words=1000, background_color="white").generate(content1list)
plt.title("Cluster: cental_bank")
plt.axis("off")
plt.imshow(wd1)
plt.show()

df2 = df[df.label==2]
content2 = df2['content']
content2list = '\t'.join([word for word in content2])
mask = np.array(image.open("currency.jpeg"))
wd2 = WordCloud(mask=mask,max_font_size=100, max_words=1000, background_color="white").generate(content2list)
plt.title("Cluster: traditional_currency")
plt.axis("off")
plt.imshow(wd2)
plt.show()   
print(df) 
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 15 01:17:49 2021

@author: haoyuwang
"""
import wordcloud
from wordcloud import WordCloud, STOPWORDS
import csv
import PIL.Image as image
import matplotlib.pyplot as plt

path="/Users/haoyuwang/Downloads/harrywang/GU/Courses/501/Cluster/"
filelocation="textdata copy.csv"




#####Read the headline column to list
my_list = []
with open((path+filelocation),'rt') as f:
    reader = csv.reader(f)
    my_list = '\t'.join([i[3] for i in reader])
(my_list)

######create a wordcloud

wd0 = WordCloud(max_font_size=50, max_words=1000, background_color="white").generate(my_list)
plt.title("text_dataset")
plt.axis("off")
plt.imshow(wd0)
plt.show()
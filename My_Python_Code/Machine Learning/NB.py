#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 12 08:46:03 2021

@author: haoyuwang
"""
import os
cwd=os.getcwd()
os.chdir('/Users/haoyuwang/Downloads/harrywang/GU/Courses/501/Discussion 6')
import pandas as pd
from pandas import DataFrame

##Tokenize and vectorize text
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
from wordcloud import WordCloud,STOPWORDS
import matplotlib.pyplot as plt

##Models import
from sklearn.model_selection import train_test_split
import random as rd
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn import tree
import graphviz
from sklearn.metrics import plot_confusion_matrix

###############################Code Starts!###############################
newsdf = pd.read_csv("News.csv",error_bad_lines=False)
newsdf.head(16)  ###df.head() cannot be printed

##Tokenize and Vectorize the Headlines
headlineLIST=[]
labelLIST=[]
for nexthead,nextlabel in zip(newsdf["Headline"],newsdf["Label"]):
    headlineLIST.append(nexthead)
    labelLIST.append(nextlabel)
    
print("The headline list is :\n")
print(headlineLIST)
print("The labek list is :\n")
print(labelLIST)

##Remove all words matching the topic
topics=["Crypto","Stock"]
#NewheadlineList=[]

#for element in headlineLIST:
#    AllWords=element.split(" ")
#    print(AllWords)

#NewwordsList=[]
#for word in AllWords:
#    print(word)
#    word=word.lower()
#    if word in topics:
#        print(word)
#    else:
#        NewwordsList.append(word)
        
        
#print(NewwordsList)
#NewWords="".join(NewwordsList)
#NewheadlineList.append(NewWords)
#print(NewheadlineList)

##New headlineList to HeadlineLISY
#HeadlineLIST=NewheadlineList
#print(HeadlineLIST)

###Build the labeled dataframe
##Vectorize
MyCountV=CountVectorizer(
        input="content",  ## because we have a csv file
        lowercase=True, 
        stop_words = "english",
        max_features=50
        )

#Fit into files
MyDTM = MyCountV.fit_transform(headlineLIST)  # create a sparse matrix
print(type(MyDTM))

ColumnNames=MyCountV.get_feature_names()
print(type(ColumnNames))

#Build the dataframe
DTDF=pd.DataFrame(MyDTM.toarray(),columns=ColumnNames)

#Convert label from list to df
labelDF=DataFrame(labelLIST,columns=['Label'])

## Check your new DF and you new Labels df:
print("Labels\n")
print(labelDF)
print("News df\n")
print(DTDF.iloc[:,0:6])

#Save the original DF
original_DTDF=DTDF
DTDF=DTDF.drop(topics,axis=1)
print(DTDF)

##Create a df with labels
dfs=[labelDF,DTDF]
FinalDF=pd.concat(dfs,axis=1,join='inner')
print(FinalDF)
 
##########################NB##########################
###Create Training and Testing Data
labelDF.to_csv("News Labels.csv")
trainDF,testDF=train_test_split(FinalDF,test_size=0.3)
print(trainDF)
print(testDF)

###Seperating labels
xtrain = trainDF.drop(["Label"],axis=1)
ytrain = trainDF["Label"].tolist()

xtest = testDF.drop(["Label"],axis=1)
ytest = testDF["Label"]
print(testDF)


###Run NB
from sklearn.naive_bayes import MultinomialNB
MyModelNB = MultinomialNB()

my_NB = MyModelNB.fit(xtrain,ytrain)
my_NB_pred = MyModelNB.predict(ytest)


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 12 08:46:03 2021

@author: haoyuwang
"""
import os
cwd=os.getcwd()
os.chdir('/Users/haoyuwang/Downloads/harrywang/GU/Courses/501/DT')
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

###Split training and testing dataset
##Make some visualization
WClist=[]
for mytopic in topics:
    tempdf = FinalDF[FinalDF['Label']==mytopic]
    print(tempdf)
    tempdf = tempdf.sum(axis=0,numeric_only=True)
    NextVarName=str("wc"+str(mytopic))
    print(NextVarName)
    NextVarName = WordCloud(width=1000, height=600, background_color="white",
                   min_word_length=4, #mask=next_image,
                   max_words=200).generate_from_frequencies(tempdf)
    WClist.append(NextVarName)

print(WClist)

#Create wordclouds
fig=plt.figure(figsize=(25, 25))
#figure, axes = plt.subplots(nrows=2, ncols=2)
NumTopics=len(topics)
for i in range(NumTopics):
    print(i)
    ax = fig.add_subplot(NumTopics,1,i+1)
    plt.imshow(WClist[i], interpolation='bilinear')
    plt.axis("off")
    plt.savefig("NewCloud.jpg")
 
##########################Create Decision Tree##########################
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


###Run DT
MyDT=DecisionTreeClassifier(criterion='gini',splitter='random',max_depth=None,min_samples_split=2,min_samples_leaf=1,min_weight_fraction_leaf=0.0,max_features=None,random_state=None,max_leaf_nodes=None,min_impurity_decrease=0.0,min_impurity_split=None,class_weight=None)
MyDT.fit(xtrain,ytrain)
tree.plot_tree(MyDT)

feature_names=xtrain.columns
Tree_Object = tree.export_graphviz(MyDT,out_file=None,feature_names=feature_names,class_names=topics,filled=True,rounded=True,special_characters=True)
graph = graphviz.Source(Tree_Object)
graph.render("MyTree3")

##########Confusion Matrix
from sklearn.metrics import confusion_matrix
ypred=MyDT.predict(xtest)
matrix = confusion_matrix(ytest, ypred)
print(matrix)
plot_confusion_matrix(MyDT,xtest,ytest)
plt.show()

Featurelmp = MyDT.feature_importances_
indices = np.argsort(Featurelmp)[::-1]
for f in range(xtrain.shape[1]):
    if Featurelmp[indices[f]]>0:
        print("%d. feature %d (%f)" % (f + 1, indices[f], Featurelmp[indices[f]]))
        print ("feature name: ", feature_names[indices[f]])

IMPORTANCE=pd.DataFrame(columns=['importance','feature'])
IMPORTANCE['importance']=MyDT.feature_importances_
IMPORTANCE['feature']=pd.DataFrame(xtrain.columns)
IMPORTANCE= IMPORTANCE.sort_values(by=['importance'],ascending=False).iloc[0:6,0:2]
plt.bar(IMPORTANCE['feature'],IMPORTANCE['importance'],width=0.6)






















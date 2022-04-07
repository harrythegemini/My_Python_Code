#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 28 23:02:35 2022

@author: haoyuwang
"""

import pandas as pd

##READ
dis = pd.read_csv('data/districts.csv')
print("ORIGINAL \n", dis.head(3))

##1. municipality info
#SPLIT Using .zip()
dis['population<500'],dis['500-1999'],dis['2000-9999'],dis['>=10000'] = zip(*dis.municipality_info.str.split(','))
dis1 = dis.drop(columns ='municipality_info')


#Remove brackets
dis1['population<500'].replace(regex = True, to_replace = r'\D', value =r'', inplace=True)
dis1['>=10000'].replace(regex = True, to_replace = r'\D', value =r'', inplace=True)
dis1.head(3)


##2. unemployment rate
#Split using .zip()
dis1['95_unemploy'],dis1['96_unemploy'] = zip(*dis1.unemployment_rate.str.split(','))
dis2 = dis1.drop(columns = 'unemployment_rate')
dis2.head(3)

#Remove bracket
#Since the number is not interger, not to use regular expression here.
dis2['95_unemploy'] = dis2['95_unemploy'].map(lambda x:str(x)[1:])
dis2['96_unemploy'] = dis2['96_unemploy'].map(lambda x:str(x)[:-1])


##3.comitted_crime
#SPlit using .zip()
dis2['95_crime'],dis2['96_crime'] = zip(*dis2.commited_crimes.str.split(','))
dis3 = dis2.drop(columns='commited_crimes')
dis3.head(3)

#Removing brackets using regular expression
dis3['95_crime'].replace(regex=True, inplace = True, to_replace =r'\D', value = r'' )
dis3['96_crime'].replace(regex=True, inplace = True, to_replace =r'\D', value = r'' )
print("AFTER \n", dis3.head(3))  
print(dis3.shape[1]) #16 columns

##MELT
dis4 = dis3.melt(id_vars=['id','name','region','population','num_cities','urban_ratio','avg_salary','entrepreneur_1000'],
                 var_name="variables", 
                 value_name='value')
print("Distric long \n",dis4.head(4))

##EXPORT TO CSV
dis4.to_csv('district_py.csv')


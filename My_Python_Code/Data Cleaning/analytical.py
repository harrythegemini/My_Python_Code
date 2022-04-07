#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 29 15:56:22 2022

@author: haoyuwang
"""
import pandas as pd
import numpy as np

#READ IN ALL DATA
account = pd.read_csv('data/accounts.csv')
district = pd.read_csv('data/districts.csv')
links = pd.read_csv('data/links.csv')
cards = pd.read_csv('data/cards.csv')
loans = pd.read_csv('loans_r.csv')
transacts = pd.read_csv('data/transactions.csv')


##1.Accounts + Districts
df = account.merge(district, how = 'left', left_on = ['district_id'], right_on=['id'],suffixes=('','_y'))
df.head()

#Drop columns
df = df[['id','date','statement_frequency','name']]
df.head()
print("Stage 1 \n", df.head())

##2.Links + Cards
LC = links.merge(cards, how = 'left', left_on=['id'], right_on=['link_id'], suffixes=['', '_y'])
LC.head()

#Drop columns
LC = LC[['id','client_id','account_id','type','type_y']]
LC.head()

#Make type_y categorical.
LC['type_y'] = LC['type_y'].fillna(0)
LC['type_y'] = np.where((LC.type_y != 0), 1, LC.type_y)
LC.head()
LC.tail()
type(LC['type_y'][1]) #This column are int.
sum(LC['type_y']) 
LC = LC.rename(columns={"type_y" : "credit_cards"})
LC = LC.drop(columns=['type','id'])
LC.head()

#Calculate number of clients and cards
#Num_customers
LC2 = LC[['account_id','client_id']]
LC3 = LC[['account_id','credit_cards']]
LC2.head()
LC3.head()

#Groupby seperately
# ref: https://stackoverflow.com/questions/19384532/get-statistics-for-each-group-such-as-count-mean-etc-using-pandas-groupby
LC2 = LC2.groupby(['account_id']).size().reset_index(name = 'num_customers')
LC2.head()
# ref: https://stackoverflow.com/questions/39922986/how-do-i-pandas-group-by-to-get-sum
LC3 = LC3.groupby(['account_id'])['credit_cards'].sum().reset_index()
LC3.head()
LC3.tail()
sum(LC3['credit_cards'])

#Merge LC2 and LC3
LC4 = LC2.merge(LC3, how = 'left', left_on='account_id', right_on='account_id', suffixes=('', '_y'))
LC4.head()

#Merge to a big table
df2 = df.merge(LC4, how = 'left', left_on='id', right_on='account_id', suffixes=('','_y'))
df2 = df2.drop(columns='account_id')
df2.head()
print("Stage 2 \n", df2.head())

##3.Loans
df3 = df2.merge(loans, how = 'left',left_on='id', right_on='account_id', suffixes=('','_y'))
df3 = df3.drop(columns=['id_y','account_id','date_y',])
df3.head()
df3.tail()

#Create column 'loan'
df3['yes_or_no'] = df3['yes_or_no'].fillna("F")
df3['yes_or_no'] = np.where(df3.yes_or_no =='X','T',df3.yes_or_no)
df3.head()

#Create 'Loan_default column'
df3['loan_default'] = df3['payment_status']
df3['loan_default'] = np.where((df3.loan_default == 'B') | (df3.loan_default == 'D'), 'T', df3.loan_default )
df3['loan_default'] = np.where((df3.loan_default == 'A') | (df3.loan_default == 'C'), 'F', df3.loan_default )
df3.head()

#Change names
df3 = df3.rename(columns={"yes_or_no":"loan", "amount":"loan_amount", "payments":"loan_payments", "term":"loan_term", "payment_status":"loan_status"})
df3.head()

#Edit 'loan_status' content
df3['loan_status'] = np.where((df3.loan_status == 'A' )|(df3.loan_status == 'B' ),'expired', df3.loan_status )
df3['loan_status'] = np.where((df3.loan_status == 'C' )|(df3.loan_status == 'D' ),'current', df3.loan_status )
print("Stage 3\n", df3.head())

##4.Transactions
transacts.head()

#Deal with Withdrawal
wd = transacts[transacts['type'] == 'debit' ]
wd.head()
wd = wd[['account_id','amount']]
grpwd = wd.groupby("account_id")
maxwd = grpwd.max()
minwd = grpwd.min()
wd2 = maxwd.reset_index()
wd2.head()
wd3 = minwd.reset_index()
wd3.head()
wd = wd2.merge(wd3,how ='left', left_on='account_id', right_on='account_id', suffixes=('','_min'))
wd = wd.rename(columns={'amount':'max_withdrawal', 'amount_min':'min_withdrawal'})
wd.head()

#Deal with balance
bl = transacts[['account_id','balance']]
grpbl = bl.groupby('account_id')
maxbl = grpbl.max()
minbl = grpbl.min()
bl2 = maxbl.reset_index()
bl3 = minbl.reset_index()

bl = bl2.merge(bl3,how ='left', left_on='account_id', right_on='account_id', suffixes=('','_min'))
bl = bl.rename(columns={"balance":"max_balance", "balance_min":"min_balance"})
bl.head()

#Create cc_payments
cd = transacts[transacts['type'] =='credit' ]
cd = cd[['account_id','type']]
cd  = cd.groupby('account_id').size().reset_index(name = 'cc_payments')
cd.head()

#Get account number
#ac = transacts[['account_id','account']]
#ac = ac.drop_duplicates() #Drop duplicates
#ac.head()

#Merge all to df4
df4 = wd.merge(bl,how ='left', left_on='account_id', right_on='account_id')
df4 = df4.merge(cd,how ='left', left_on='account_id', right_on='account_id')
print("Stage 4 \n", df4.head())

##5.Merge all to a final df5
df5 = df3.merge(df4,how ='left', left_on='id', right_on='account_id')
df5 = df5.drop(columns='account_id')

#Rename columns
df5 = df5.rename(columns={"id":"account_id", "date":"open_date","name":"district_name"})

#Reorder
df5 = df5[["account_id","open_date","statement_frequency","num_customers","credit_cards","loan","loan_amount","loan_payments","loan_term","loan_status","loan_default","max_withdrawal","min_withdrawal","cc_payments","max_balance","min_balance"]]
print("Stage 5 \n", df5.head())

##Export to csv
df5.to_csv('analytical_py.csv')






import pandas as pd
import numpy as np

##READ
loan = pd.read_csv('data/loans.csv')
print("ORGINAL \n", loan.head(3))

##MELT
loan1 = loan.melt(id_vars=['id','account_id','date','amount','payments'], var_name='loan_detail',value_name='yes_or_no')
print("AFTER MELT \n",loan1.head(3))

##SPLIT
loan2 = (loan1.assign(
    term = lambda x: x.loan_detail.str[0:1].astype(str),
    payment_status = lambda x: x.loan_detail.str[3].astype(str))
    .drop('loan_detail', axis=1))
print("AFTER SPLIT \n", loan2.head(3))

##REMOVE USELESS ROWS
loan3  = loan2[loan2['yes_or_no']=='X']
print("AFTER REMOVING \n", loan3.head())
loan3.info()

##EXPORT TO CSV
loan3.to_csv('loans_py.csv')


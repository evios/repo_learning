#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 10 10:46:52 2019

@author: ei
"""

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
# %matplotlib inline
import seaborn as sns



"""
print("Setup Complete")


def test_func(X):
    return X+1

scores = {X: test_func(X) for X in [1,2,3]}

df = pd.DataFrame(scores, index = ['One', 'Two', 'Three'])
sns.lineplot(data=df)
sns.heatmap(data=df)
sns.barplot(data=df)
#plt.show()

sns.lineplot(data=df.T)
sns.heatmap(data=df.T)
sns.barplot(data=df.T)

museum= {'2018-07-01':2620, '2018-08-01':2409, '2018-09-01':2146, '2018-10-01':2364}
mdata=pd.DataFrame(museum, index = ['Chinese American Museum'])
mdataT=mdata.T
"""

#filename='./goodreads_books.csv'
books=pd.read_csv('./goodreads_books.csv', index_col='bookID', skiprows=1, names=['bookID', 'title','authors','average_rating','isbn','isbn13','language_code','# num_pages','ratings_count','text_reviews_count'])

#goodreads_books.head()

#books.average_rating=books.average_rating.astype(float)
## ValueError: could not convert string to float: ' Jr.-Sam B. Warner'

books.average_rating=pd.to_numeric(books.average_rating,errors='coerce')
books['average_rating']=pd.to_numeric(books['average_rating'],errors='coerce')

sns.lineplot(data=books['average_rating'])

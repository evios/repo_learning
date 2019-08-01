""" Templates for DS/ML pipeline
Oftenly used blocks of code
"""
# import libraries 
import pandas as pd

#data load
pd.read_csv('./goodreads_books.csv')
>>> with open('workfile') as f:
...     read_data = f.read()
>>> f.closed
True
#close file

#??VectorAssembler (X_features separation from y)
#	https://www.youtube.com/watch?v=oDTJxEl95Go&t=1120


#correlation
#	plot

#clean
#	impute
#	drop
#	fillna


#categorical to numberical

#normalize
#	StandardScaler

#split train-test

#model selection
#	fit
#predict

#evaluate
#???	evaluator - BinaryClassificationEvaluator

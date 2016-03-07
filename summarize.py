# -*- coding: utf-8 -*-
"""
Created on Sun Mar  6 17:40:32 2016

@author: Dooshkukakoo
"""

import pickle
import random

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from nltk.corpus import stopwords
import matplotlib.pyplot as plt

btr = pickle.load(open("dict-of-business-to-reviews.p", "rb"))
docnames = ["ASC", "Burger King", "McDonald's", "Hunter Farm", "PCR"]



from reduction import *
reduction = Reduction()
text = ''
for review in btr["Burger King"][0:5]:
    print(review)
    text += review;
reduction_ratio = 0.3
reduced_text = reduction.reduce(text, reduction_ratio)
print(reduced_text)
print("end")
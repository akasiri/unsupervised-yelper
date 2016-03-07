########
### Code found online from http://www.puffinwarellc.com
### Uses scipy for LSA; the code I wrote below uses sklearn
########
#
#from numpy import zeros
#from scipy.linalg import svd
##following needed for TFIDF
#from math import log
#from numpy import asarray, sum
#
#titles = ["The Neatest Little Guide to Stock Market Investing",
#          "Investing For Dummies, 4th Edition",
#          "The Little Book of Common Sense Investing: The Only Way to Guarantee Your Fair Share of Stock Market Returns",
#          "The Little Book of Value Investing",
#          "Value Investing: From Graham to Buffett and Beyond",
#          "Rich Dad's Guide to Investing: What the Rich Invest in, That the Poor and the Middle Class Do Not!",
#          "Investing in Real Estate, 5th Edition",
#          "Stock Investing For Dummies",
#          "Rich Dad's Advisors: The ABC's of Real Estate Investing: The Secrets of Finding Hidden Profits Most Investors Miss"
#          ]
#stopwords = ['and','edition','for','in','little','of','the','to']
#ignorechars = ''',:'!'''
#
#class LSA(object):
#    def __init__(self, stopwords, ignorechars):
#        self.stopwords = stopwords
#        self.ignorechars = ignorechars
#        self.wdict = {}
#        self.dcount = 0        
#    def parse(self, doc):
#        words = doc.split();
#        for w in words:
#            w = w.lower().translate(None, self.ignorechars)
#            if w in self.stopwords:
#                continue
#            elif w in self.wdict:
#                self.wdict[w].append(self.dcount)
#            else:
#                self.wdict[w] = [self.dcount]
#        self.dcount += 1      
#    def build(self):
#        self.keys = [k for k in self.wdict.keys() if len(self.wdict[k]) > 1]
#        self.keys.sort()
#        self.A = zeros([len(self.keys), self.dcount])
#        for i, k in enumerate(self.keys):
#            for d in self.wdict[k]:
#                self.A[i,d] += 1
#    def calc(self):
#        self.U, self.S, self.Vt = svd(self.A)
#    def TFIDF(self):
#        WordsPerDoc = sum(self.A, axis=0)        
#        DocsPerWord = sum(asarray(self.A > 0, 'i'), axis=1)
#        rows, cols = self.A.shape
#        for i in range(rows):
#            for j in range(cols):
#                self.A[i,j] = (self.A[i,j] / WordsPerDoc[j]) * log(float(cols) / DocsPerWord[i])
#    def printA(self):
#        print ('Here is the count matrix')
#        print (self.A)
#    def printSVD(self):
#        print ('Here are the singular values')
#        print (self.S)
#        print ('Here are the first 3 columns of the U matrix')
#        print (-1*self.U[:, 0:3])
#        print ('Here are the first 3 rows of the Vt matrix')
#        print (-1*self.Vt[0:3, :])
#
#mylsa = LSA(stopwords, ignorechars)
#for t in titles:
#    mylsa.parse(t)
#mylsa.build()
#mylsa.printA()
#mylsa.calc()
#mylsa.printSVD()

"""
set up the "pipeline" so that we have the topic model and then test a 
smaller selection of reviews on it

To get relevant scentences:
> model based on noun phrases
> have a dict mapping the noun phrases to sentences they came from
> then get the top noun phrases for a topic and use their sentences for our 
topic summary
"""

import pickle
import random

# for consistent testing
random.seed(1532525625823)

raw_data = pickle.load(open("pickles/list-of-reviews.p", "rb"))

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from nltk.corpus import stopwords

count_vect = CountVectorizer(stop_words=set(stopwords.words('english')))
train_counts = count_vect.fit_transform(random.sample(raw_data, 30000))

raw_data = None
btr = pickle.load(open("pickles/dict-of-business-to-reviews.p", "rb"))

test_counts = count_vect.transform(btr["Appliance Service Center"])

tfidf_transformer = TfidfTransformer()
train_tfidf = tfidf_transformer.fit_transform(train_counts)
test_tfidf = tfidf_transformer.transform(test_counts)

dtm = train_tfidf
dtm_test = test_tfidf

vocab = count_vect.get_feature_names()

# pickle.dump(dtm, open("term-doc-matrix.p", "wb"))
# dtm = pickle.load(open("term-doc-matrix.p", "rb"))
    
import numpy as np  # a conventional alias
import sklearn.feature_extraction.text as text

from sklearn import decomposition

num_topics = 60
num_top_words = 20

lsa = decomposition.TruncatedSVD(n_components=num_topics, random_state=1)

## this next step may take some time
#doctopic = lsa.fit_transform(dtm)
#doctopic = doctopic / np.sum(doctopic, axis=1, keepdims=True)
#
#
## print words associated with topics
#topic_words = []
#for topic in lsa.components_:
#    word_idx = np.argsort(topic)[::-1][0:num_top_words]
#    topic_words.append([vocab[i] for i in word_idx])
#
#
#print(topic_words)
#print(word_idx)
#
#
#print()
#print()
#for t in range(len(topic_words)):
#   print("Topic {}: {}".format(t, ' '.join(topic_words[t][:10])))
#   
#result = lsa.transform(dtm_test)
#
#
## Find the top topics for the restaurant given above
#m = []
#for i in range(num_topics):
#    m.append(0)
#    
#for i in result:
#    for j in range(num_topics):
#        m[j] += i[j]
#
#top5 = [(0,0),(0,0),(0,0),(0,0),(0,0)]
#
#for i in range(num_topics):
#    if m[i] > top5[4][1]:
#        top5[4] = (i, m[i])
#        top5.sort(reverse=True)
#
#for (t,p) in top5:
#    print("Topic {}: {}".format(t, ' '.join(topic_words[t][:10])))


"""
novel_names = []

for fn in filenames:
    basename = os.path.basename(fn)
    name, ext = os.path.splitext(basename)
    name = name.rstrip('0123456789')
    novel_names.append(name)

# turn this into an array so we can use NumPy functions
novel_names = np.asarray(novel_names)
doctopic_orig = doctopic.copy()

# use method described in preprocessing section
num_groups = len(set(novel_names))

doctopic_grouped = np.zeros((num_groups, num_topics))

for i, name in enumerate(sorted(set(novel_names))):
    doctopic_grouped[i, :] = np.mean(doctopic[novel_names == name, :], axis=0)

doctopic = doctopic_grouped
"""
    
"""
throw out short reviews
throw out words used in less than 20 reviews

look for bi-grams and tri-grams that are noun phrases: more accurate for this type of work

use the topics of a document as features of the document and compare it to ratings
"""

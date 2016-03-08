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

# load the list of all reviews
raw_data = pickle.load(open("pickles/list-of-reviews.p", "rb"))

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from nltk.corpus import stopwords
import matplotlib.pyplot as plt

# set up a count vectorizer that removes english stopwords when building a term-doc matrix
count_vect = CountVectorizer(stop_words=set(stopwords.words('english')))

# build the term frequency per document matrix from a random sublist of 30,000 documents
train_counts = count_vect.fit_transform(random.sample(raw_data, 30000))

# reset the raw_data to clear space in memory
raw_data = 0

# load business to list of reviews for that business dictionary
#btr = pickle.load(open("pickles/dict-of-business-to-reviews.p", "rb"))

btr = pickle.load(open("dict-of-business-to-reviews.p", "rb"))
docnames = ["Appliance Service Center", "Burger King", "McDonald's", "Hunter Farm", "Panda Chinese Restaurant"]
docnames.sort()
# johnathan's thing
test_counts = count_vect.transform(btr["Appliance Service Center"] + btr["Burger King"] + btr["Hunter Farm"] + btr["McDonald's"] + btr["Panda Chinese Restaurant"])

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

num_topics = 20
num_top_words = 15

nmf = decomposition.NMF(n_components=num_topics, random_state=1)

# this next step may take some time
doctopic = nmf.fit_transform(dtm)
doctopic = doctopic / np.sum(doctopic, axis=1, keepdims=True)
#print(doctopic.shape)
#
#print("components")
#print(nmf.components_)
# print words associated with topics
topic_words = []
for topic in nmf.components_:
    word_idx = np.argsort(topic)[::-1][0:num_top_words]
    topic_words.append([vocab[i] for i in word_idx])

"""
print(topic_words)
print(word_idx)
"""

print()
print()
for t in range(len(topic_words)):
   print("Topic {}: {}".format(t+1, ' '.join(topic_words[t][:10])))
   

result = nmf.transform(dtm_test)

# Find the top topics for the restaurant given above
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
#    print("Topic {}: {}".format(t+1, ' '.join(topic_words[t][:10])))
    

doctopic = nmf.transform(dtm_test)
doctopic = doctopic / np.sum(doctopic, axis=1, keepdims=True)

#print(dtm_test)
#print(doctopic)

# turn this into an array so we can use NumPy functions
docnames = np.asarray(docnames)

doctopic_orig = doctopic.copy()

# use method described in preprocessing section
num_groups = len(set(docnames))

doctopic_grouped = np.zeros((num_groups,num_topics))

#print("i, ")
#print(doctopic_grouped[1, :])

for i, name in enumerate(sorted(set(docnames))):
    doctopic_grouped[i, :] = np.mean(doctopic[docnames == name, :], axis=0)

doctopic = doctopic_grouped
#print("doctopic_grouped")
#print(doctopic)
businesses = sorted(set(docnames))
print("businesses")
print(businesses)
print("Top NMF topics in...")

d = dict()
for business in docnames:
    d[business] = [];
    
for i in range(len(doctopic)):
    top_topics = np.argsort(doctopic[i,:])[::-1][0:3]
    top_topics_str = ' '.join(str(t+1) for t in top_topics)
    print("{}: {}".format(businesses[i], top_topics_str))
    
    words_to_look_for = []
    for t in top_topics:
        for word in topic_words[t]:
            words_to_look_for.append(word)
            
    print(words_to_look_for)            
    for review in btr[businesses[i]]:
        count = 0
        for word in words_to_look_for:
            if (word in review):
#                print(word)
                count = count + 1
                continue
            if(count/len(review.split()) > 0.125):
                d[businesses[i]].append(review)
                break;

['love', 'always', 'never', 'favorite', 'time', 'come', 'every', 'get', 'fresh', 
'location', 'times', 'usually', 'family', 'years', 'make', 'food', 'service', 
'restaurant', 'mexican', 'excellent', 'fast', 'quality', 'bad', 'buffet', 'fresh', 
'eat', 'chinese', 'slow', 'atmosphere', 'delicious', 'chicken', 'ordered', 'salad', 
'delicious', 'sauce', 'cheese', 'lunch', 'rice', 'fried', 'menu', 'soup', 'sandwich', 
'beef', 'meal', 'pork']
#for k,v in d.items():
#    print(k)
#    print(len(v))
#    print(v)
#    print()

    
#print(doctopic.shape)
N, K = doctopic.shape  # N documents, K topics

ind = np.arange(N)  # the x-axis locations for the novels

#print("ind")
#print(ind)

width = 0.5  # the width of the barsb
plots = []

height_cumulative = np.zeros(N)

for k in range(K):
    color = plt.cm.coolwarm(k/K, 1)
    if k == 0:
        p = plt.bar(ind, doctopic[:, k], width, color=color)
    else:
        p = plt.bar(ind, doctopic[:, k], width, bottom=height_cumulative, color=color)
    height_cumulative += doctopic[:, k]
    plots.append(p)

plt.ylim((0, 1))  # proportions sum to 1, so the height of the stacked bars is 1

plt.ylabel('Topics')


plt.title('Topics in businesses')
docnames = ["ASC", "Burger King", "McDonald's", "Hunter Farm", "PCR"]
docnames.sort()
plt.xticks(ind+width/2, docnames)

plt.yticks(np.arange(0, 1, 10))

topic_labels = ['Topic {}'.format(k+1) for k in range(K)]

#box = plt.get_position()
#plt.set_position([box.x0, box.y0, box.width * 0.8, box.height])
# see http://matplotlib.org/api/pyplot_api.html#matplotlib.pyplot.legend for details
# on making a legend in matplotlib

plt.legend([p[0] for p in plots], topic_labels, bbox_to_anchor=(1.3, 1.2))
plt.show()

docnames = ["Appliance Service Center", "Burger King", "McDonald's", "Hunter Farm", "Panda Chinese Restaurant"]
docnames.sort()

from reduction import *

reduction_ratios = []
for business in docnames:
    aggregate_reviewtext = ''
    for review in d[business]:
        aggregate_reviewtext += review
#    reduction_ratio = len(d[business])/(len(btr[business]) + len(aggregate_reviewtext))
    reduction = Reduction()
    reduced_text = reduction.reduce(aggregate_reviewtext, 0.125)
    print("Summary of " + business)
    print(reduced_text)

print("end")
print("done")

#mallet_vocab = []
#
#word_topic_counts = []
#
#with open("/tmp/word-topic.txt") as f:
#for line in f:
#    _, word, *topic_count_pairs = line.rstrip().split(' ')
#    topic_count_pairs = [pair.split(':') for pair in topic_count_pairs]
#    mallet_vocab.append(word)
#    counts = np.zeros(num_topics)
#    for topic, count in topic_count_pairs:
#        counts[int(topic)] = int(count)
#    word_topic_counts.append(counts)
#
#
#In [45]: word_topic = np.array(word_topic_counts)
#
#In [46]: word_topic.shape
#Out[46]: (21988, 5)



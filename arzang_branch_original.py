import pickle
import random
import re

count_vect = None
tfidf_transformer = None

def extract_tfidf_dtm(documents, tokenizer):
    global count_vect
    global tfidf_transformer
    
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.feature_extraction.text import TfidfTransformer
    from nltk.corpus import stopwords

    count_vect = CountVectorizer(strip_accents="unicode", tokenizer = tokenizer, stop_words=set(stopwords.words('english')))
    count_matrix = count_vect.fit_transform(documents)

    tfidf_transformer = TfidfTransformer()
    tfidf_matrix = tfidf_transformer.fit_transform(count_matrix)
    
    return tfidf_matrix


## for consistent testing
#random.seed(1532525625823)
#
#raw_data = pickle.load(open("pickles/list-of-reviews.p", "rb"))
#documents = random.sample(raw_data, 30000)
#
#import advanced_parsing
#dtm = extract_tfidf_dtm(documents, advanced_parsing.extract_np_tokens)

count_vect = pickle.load(open("pickles/np-30000-count-vect.p", "rb"))
tfidf_transformer = pickle.load(open("pickles/np-30000-tfidf-trans.p", "rb"))

dtm = pickle.load(open("pickles/np-30000-dtm.p", "rb"))


import numpy as np  # a conventional alias
import sklearn.feature_extraction.text as text

from sklearn import decomposition

num_topics = 60
num_top_words = 20

lda = decomposition.LatentDirichletAllocation¶(n_topics=num_topics, random_state=1)

# this next step may take some time
doctopic = nmf.fit_transform(dtm)
doctopic = doctopic / np.sum(doctopic, axis=1, keepdims=True)


# print words associated with topics
topic_words = []
for topic in nmf.components_:
    word_idx = np.argsort(topic)[::-1][0:num_top_words]
    topic_words.append([vocab[i] for i in word_idx])


print(topic_words)
print(word_idx)
#
#
#print()
#print()
#for t in range(len(topic_words)):
#   print("Topic {}: {}".format(t, ' '.join(topic_words[t][:10])))
#   
#
#result = nmf.transform(dtm_test)
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
#print()
#print()
#for (t,p) in top5:
#    print("Topic {}: {}".format(t, ' '.join(topic_words[t][:10])))
#    
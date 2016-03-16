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

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from nltk.corpus import stopwords

import numpy as np  # a conventional alias
import sklearn.feature_extraction.text as text
from sklearn import decomposition

def get_reviews_for_business_name(name):
    btr = pickle.load("dict-of-business-to-reviews.p", "rb")
    return btr["name"]


class TopicModeler:
    
    seed = 1532525625823
    
    count_vect = None
    tfidf_transformer = None
    vocab = None
    model = None
    
    default_num_topics = 60
    
    def extract_features(self, documents : [str]):  
        """
        Builds tfidf matrix to be used when making the topic modelers
        """
        
        self.count_vect = CountVectorizer(stop_words=set(stopwords.words('english')))
        train_counts = self.count_vect.fit_transform(documents)
                
        self.tfidf_transformer = TfidfTransformer()
        train_tfidf = self.tfidf_transformer.fit_transform(train_counts)
        
        self.vocab = self.count_vect.get_feature_names()
        return train_tfidf # document-term matrix holding tfidf values
    
    def train_NMF_model(self, dtm : "doc-term tfidf matrix", num_topics = default_num_topics):
        self.model = decomposition.NMF(n_components=num_topics, random_state=1)
        self.model.fit_transform(dtm)
        
    def train_LSA_model(self, dtm : "doc-term tfidf matrix", num_topics = default_num_topics):
        self.model = decomposition.TruncatedSVD(n_components=num_topics, random_state=1)
        self.model.fit_transform(dtm)
        
    def train_LDA_model(self, dtm : "doc-term tfidf matrix", num_topics = default_num_topics):
        self.model = decomposition.LatentDirichletAllocation(n_topics=num_topics, random_state=1)
        self.model.fit_transform(dtm)
     
    def build_test_matrix(self, documents : [str]):
        try:
            test_counts = self.count_vect.tranform(documents)
            test_tfidf = self.tfidf_transformer.transform(test_counts)
        except:
            print("Error. Try calling extract_features() before build_test_matrix(). The CountVectorier and TfidfTransformer need to be build before use in build_test_matrix().")
        
        return test_tfidf
    
    def predict_top_n_topics(self, busi_dtm, n : "num topics to return"):
        topic_matrix = self.model.transform(busi_dtm)
        
        
    def gather_topic_words(self, num_top_words = 20) -> [[str]]:
        """
        Returns a list of words associated with each topic.
        Should this just make an object variable instead of returning the words?
        """
        try:
            # print words associated with topics
            topic_words = []
            for topic in self.model.components_:
                word_idx = np.argsort(topic)[::-1][0:num_top_words]
                topic_words.append([self.vocab[i] for i in word_idx])
        except:
            print("Error. Try calling one of the train_X_model() methods before calling gather_topic_words().")
            
        return topic_words
    
    def print_topics(self, model, topic_words : [[str]], number_of_words : int):
        """
        Neatly prints out topics and the words associated with them
        """
        
        print()
        for t in range(len(topic_words)):
           print("Topic {}: {}".format(t, ' '.join(topic_words[t][:number_of_words])));
        print()
        
    def pipeline(self, name_of_target_business : str, number_of_documents : int):
        # for consistent testing
        random.seed(self.seed)
        
        raw_data = pickle.load(open("pickles/list-of-reviews.p", "rb"))
        documents = random.sample(raw_data, number_of_documents)
        
        dtm = self.extract_features(documents)
        self.train_NMF_model(dtm)
        
        target_reviews = get_reviews_for_business_name(name_of_target_business)
        dtm_test = self.build_test_matrix(target_reviews)
        self.predict_top_n_topics()
        


result = nmf.transform(dtm_test)

# Find the top topics for the restaurant given above
m = []
for i in range(num_topics):
    m.append(0)
    
for i in result:
    for j in range(num_topics):
        m[j] += i[j]

top5 = [(0,0),(0,0),(0,0),(0,0),(0,0)]

for i in range(num_topics):
    if m[i] > top5[4][1]:
        top5[4] = (i, m[i])
        top5.sort(reverse=True)

print()
print()
for (t,p) in top5:
    print("Topic {}: {}".format(t, ' '.join(topic_words[t][:10])))
    
    
if __name__ == "__main__":
    tm = TopicModeler()
    tm.pipeline("Cindy Esser's Floral Shop", 60)
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
import matplotlib.pyplot as plt

import numpy as np  # a conventional alias
import sklearn.feature_extraction.text as text

from sklearn import decomposition

from reduction import *

import warnings
warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning) 

# for consistent testing
random.seed(1532525625823)

###################
# Helper functions #
###################
def load_reviews():
    """
    Loads the list of the Yelp reviews in the Yelp data set.

    Returns
    -------
    list
        A very large list of  2,225,213 reviews.

    Example
    -------
    >>> reviews = load_reviews()
    >>> reviews[:2]
    ['Mr Hoagie is an institution. Walking in, it does seem like a throwback to 30 years ago, old fashioned menu board, booths out of the 70s, and a large selection of food. Their speciality is the Italian Hoagie, and it is voted the best in the area year after year. I usually order the burger, while the patties are obviously cooked from frozen, all of the other ingredients are very fresh. Overall, its a good alternative to Subway, which is down the road.', 
    "Excellent food. Superb customer service. I miss the mario machines they used to have, but it's still a great place steeped in tradition."]
    """
    reviews = pickle.load(open("pickles/list-of-reviews.p", "rb"))
    return reviews

def load_business_to_reviews():
    """
    Loads the list of the Yelp businesses to their reviews

    Returns
    -------
    dict
        A very large dictionary that maps 57,721 businesses to their reviews.
    """
    btr = pickle.load(open("pickles/dict-of-business-to-reviews.p", "rb"))
    return btr

###################
# Main functions #
###################
def extract_text_features(train_data, test_data):
    """
    Returns one types of training and test data features.
        1) Term Frequency times Inverse Document Frequency (tf-idf): X_train_tfidf, X_test_tfidf

    Parameters
    ----------
    train_data : List[str]
        Training data in list. Will only take 30000 reviews for efficiency purposes

    test_data : List[str]
        Test data in list

    Returns
    -------
    Tuple(scipy.sparse.csr.csr_matrix,.., list)
        Returns X_train_tfidf, X_test_tfidf, vocab as a tuple.
    """
    
    # set up a count vectorizer that removes english stopwords when building a term-doc matrix
    count_vect = CountVectorizer(stop_words=set(stopwords.words('english')))
    # build the term frequency per document matrix from a random sublist of 30,000 documents
    train_counts = count_vect.fit_transform(random.sample(train_data, 30000))
    test_counts = count_vect.transform(test_data)
    tfidf_transformer = TfidfTransformer()

    train_tfidf = tfidf_transformer.fit_transform(train_counts)
    test_tfidf = tfidf_transformer.transform(test_counts)
    
    vocab = count_vect.get_feature_names()
    
    return (train_tfidf, test_tfidf, vocab)

def fit_and_predict_NMF(num_topics, num_top_words, vocab, dtm_train, dtm_test):
    """
    Fit the NMF topic modeling to the training document term matrix.
    Using the generated topics, map test document term matrix to 
    document to topic matrix. Also return topic words.


    Parameters
    ----------
    num_topics: int
        number of topics NMF decomposition should generate
    num_top_words: int
        number of topic words stored in topic_words list
    vocab: set
        set of unique terms in the reviews
    dtm_train: scipy sparse matrix
        Data for training (matrix with features, e.g. BOW or tf-idf)
    dtm_test: scipy sparse matrix
        Data for testing and used for 'prediction' (matrix with features, e.g. BOW or tf-idf)

    Returns
    -------
    Tuple(numpy.ndarray, set)
        Returns doctopic, topic_words as a tuple
    """
    nmf = decomposition.NMF(n_components=num_topics, random_state=1)
    nmf.fit(dtm_train)
    doctopic = nmf.transform(dtm_test)
    #scale the document-component matrix such that the component values associated with each document sum to one
    doctopic = doctopic / np.sum(doctopic, axis=1, keepdims=True)
    
    topic_words = []
    for topic in nmf.components_:
        word_idx = np.argsort(topic)[::-1][0:num_top_words]
        topic_words.append([vocab[i] for i in word_idx])
        
    return (doctopic, topic_words)
    
def fit_and_predict_LDA(num_topics, num_top_words, vocab, dtm_train, dtm_test):
    """
    Fit the LDA topic modeling to the training document term matrix.
    Using the generated topics, map test document term matrix to 
    document to topic matrix. Also return topic words.


    Parameters
    ----------
    num_topics: int
        number of topics NMF decomposition should generate
    num_top_words: int
        number of topic words stored in topic_words list
    vocab: set
        set of unique terms in the reviews
    dtm_train: scipy sparse matrix
        Data for training (matrix with features, e.g. BOW or tf-idf)
    dtm_test: scipy sparse matrix
        Data for testing and used for 'prediction' (matrix with features, e.g. BOW or tf-idf)

    Returns
    -------
    Tuple(numpy.ndarray, set)
        Returns doctopic, topic_words as a tuple
    """
    lda = decomposition.LatentDirichletAllocation(n_topics=num_topics, random_state=1)
    lda.fit(dtm_train)
    doctopic = lda.transform(dtm_test)
    #scale the document-component matrix such that the component values associated with each document sum to one
    doctopic = doctopic / np.sum(doctopic, axis=1, keepdims=True)
    
    topic_words = []
    for topic in lda.components_:
        word_idx = np.argsort(topic)[::-1][0:num_top_words]
        topic_words.append([vocab[i] for i in word_idx])
    
#    for t in range(len(topic_words)):
#        print("Topic {}: {}".format(t+1, ' '.join(topic_words[t][:10])))
        
    return (doctopic, topic_words)
   
def map_topics_to_businesses(docnames, num_topics, doctopic):
    """
    Maps the businesses to their topics based on their reviews.
    Note that the parameter doctopic is a review to topic matrix.
    The returned doctopic is a business to topic matrix.

    Parameters
    ----------
    doctopic : numpy.ndarray
        A document to topic matrix each cell represents the probability of topics with reviews
    docnames: list
        A list of businesses should be the same as businesses in doctopic
    num_topics: int
        Number of topics found by decomposition earlier

    Returns
    -------
    numpy.ndarray
        Returns a business to topic matrix called doctopic
    """
    # turn this into an array so we can use NumPy functions
    docnames = np.asarray(docnames)
    
    # use method described in preprocessing section
    num_groups = len(set(docnames))
    
    doctopic_grouped = np.zeros((num_groups,num_topics))
    
    #iterates through each business and adds the probability of the topic to the business
    for i, name in enumerate(sorted(set(docnames))):
        doctopic_grouped[i, :] = np.sum(doctopic[docnames == name, :], axis=0)
    
    doctopic = doctopic_grouped
    
    return doctopic

def find_relevant_reviews(doctopic, docnames, btr, top_topics_count, topic_words, threshold):
    """
    Go through reviews of each restaurant and eliminate reviews that do not
    have a weighted term frequency greater than the threshold

    Parameters
    ----------
    doctopic : numpy.ndarray
        A document to topic matrix each cell represents the probability of topics within the business
    docnames: list
        A list of businesses should be the same as businesses in doctopic
    btr: dict
        Business to reviews dictionary
    top_topics_count: int
        Number of top topics to look at for each business
    topic_words: list [list]
        A list of lists where each inner list holds the topic_words for a topic. The index should be the topic number minus 1. 
    threshold: float
        The threshold the weighted term frequency a review has to surpass to be added to returned dict

    Returns
    -------
    dict(key: business, value: list of reviews)
        Returns businesses to most relevant reviews dict according to x top_topics
    """
    d = dict()
    for business in docnames:
        d[business] = [];
#    print("Top NMF topics in...")
    #for each business find the relevant words to look for in a review
    for i in range(len(doctopic)):
        #retrieve x amount of topics from the business sorted in descending order
        top_topics = np.argsort(doctopic[i,:])[::-1][0:top_topics_count]
    #    top_topics_str = ' '.join(str(t+1) for t in top_topics)
    #    print("{}: {}".format(businesses[i], top_topics_str))
        
        #
        words_to_look_for = []
        for t in top_topics:
            for word in topic_words[t]:
                words_to_look_for.append(word)
                 
        for review in btr[docnames[i]]:
            count = 0
            for word in words_to_look_for:
                #if review has a weighted term frequency for the relevant word > threshold add to dict
                if(count/len(review.split()) > threshold):
                    d[docnames[i]].append(review)
                    break
                #add one to term frequency
                if (word in review):
                    count = count + 1
    return d


# =================================================================
#                            PLOTTING
# =================================================================

def plot_businesses_histogram(doctopic, docnames, legend=False):
    """
    Plot a histogram of the businesses to the probabilities of their topics

    Parameters
    ----------
    doctopic : numpy.ndarray
        A document to topic matrix each cell represents the probability of topics within the business
    docnames: list
        A list of businesses should be the same as businesses in doctopic
    legend: boolean
        Boolean to track if user wants to display a legend or not for the topics
    """    
    #print(doctopic.shape)
    N, K = doctopic.shape  # N documents, K topics
    
    ind = np.arange(N)  # the x-axis locations for the novels
    
    width = 0.5  # the width of the bars
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
    
    #shorten long business names to fit histogram nicely
    docnames_short = []
    for i in range(len(docnames)):
        if(len(docnames[i]) > 10):
            docnames_short.append(docnames[i][:11])
        else:
            docnames_short.append(docnames[i])
            
    plt.title('Topics in businesses')
    plt.xticks(ind+width/2, docnames_short)
    
    plt.yticks(np.arange(0, 1, 10))
    
    topic_labels = ['Topic {}'.format(k+1) for k in range(K)]
    
    if(legend):
        plt.legend([p[0] for p in plots], topic_labels, bbox_to_anchor=(1.3, 1.2))
    
    plt.show()
    

# =================================================================
#                         SUMMARIZATION
# =================================================================

def print_summary(docnames, dict_businesses):
    """
    Print summary of the relevant reviews of a business using Reduction github code.
    Importance of sentences determined using TextRank.

    Parameters
    ----------
    docnames: list
        A list of businesses should be the same as businesses in doctopic
    dict_businesses: dict
        Maps business names to their 'relevant' reviews
    """     
    for business in docnames:
        aggregate_reviewtext = ''
        for review in dict_businesses[business]:
            aggregate_reviewtext += review
    #    reduction_ratio = len(d[business])/(len(btr[business]) + len(aggregate_reviewtext))
            
    #    print("=== " + aggregate_reviewtext + " ===")
        
        if (len(aggregate_reviewtext) > 0): # A business needs to have at least one review to be summarized
            reduction = Reduction()
            
            ratio = 100 / len(aggregate_reviewtext)
            if (ratio > 0.62): #if document has a ratio greater than the golden ratio set it equal to golden ratio
                ratio = 0.62
            reduced_text = reduction.reduce(aggregate_reviewtext, ratio)
        else:
            reduced_text = "N/A"
        print("Summary of " + business)
        print(reduced_text)

def pipeline_NMF(test_data, num_topics, num_top_words, docnames, top_topics_count, threshold, legend=False):
    """
    Pipeline for all our functions defined above for NMF topic modeling.

    Parameters
    ----------
    test_data : List[str]
        Test data in list
    num_topics: int
        number of topics NMF decomposition should generate
    num_top_words: int
        number of topic words stored in topic_words list
    docnames: list
        A list of businesses names
    top_topics_count: int
        Number of top topics to look at for each business
    threshold: float
        The threshold the weighted term frequency a review has to surpass to be added to returned dict
    legend: boolean
        Boolean to track if user wants to display a legend or not for the topics
    """     
    reviews = load_reviews()
    btr = load_business_to_reviews()
    
    #hard_coded to test
    test_data = btr["Appliance Service Center"] + btr["Burger King"] + btr["Hunter Farm"] + btr["McDonald's"] + btr["Panda Chinese Restaurant"] 
    
    dtm_train, dtm_test, vocab = extract_text_features(reviews, test_data)
    
    doctopic, topic_words = fit_and_predict_NMF(num_topics, num_top_words, vocab, dtm_train, dtm_test)
    
#    for t in range(len(topic_words)):
#        print("Topic {}: {}".format(t+1, ' '.join(topic_words[t][:10])))
        
    doctopic = map_topics_to_businesses(docnames, num_topics, doctopic)

    btr_relevant = find_relevant_reviews(doctopic, docnames, btr, top_topics_count, topic_words, threshold)
    
    plot_businesses_histogram(doctopic, docnames)
    
    print_summary(docnames, btr_relevant)
    
def pipeline_LDA(test_data, num_topics, num_top_words, docnames, top_topics_count, threshold, legend=False):
    """
    Pipeline for all our functions defined above for LDA topic modeling.

    Parameters
    ----------
    test_data : List[str]
        Test data in list
    num_topics: int
        number of topics NMF decomposition should generate
    num_top_words: int
        number of topic words stored in topic_words list
    docnames: list
        A list of businesses names
    top_topics_count: int
        Number of top topics to look at for each business
    threshold: float
        The threshold the weighted term frequency a review has to surpass to be added to returned dict
    legend: boolean
        Boolean to track if user wants to display a legend or not for the topics
    """     
    reviews = load_reviews()
    btr = load_business_to_reviews()

    #hard_coded to test
    test_data = btr["Appliance Service Center"] + btr["Burger King"] + btr["Hunter Farm"] + btr["McDonald's"] + btr["Panda Chinese Restaurant"] 
    
    dtm_train, dtm_test, vocab = extract_text_features(reviews, test_data)

    doctopic, topic_words = fit_and_predict_LDA(num_topics, num_top_words, vocab, dtm_train, dtm_test)
        
    doctopic = map_topics_to_businesses(docnames, num_topics, doctopic)

    btr_relevant = find_relevant_reviews(doctopic, docnames, btr, top_topics_count, topic_words, threshold)
    
    plot_businesses_histogram(doctopic, docnames)
    
    print_summary(docnames, btr_relevant)
    
    
if __name__ == '__main__':
    num_topics = 60
    num_top_words = 15
    docnames = ["Appliance Service Center", "Burger King", "McDonald's", "Hunter Farm", "Panda Chinese Restaurant"]
    docnames.sort()
    top_topics_count = 5
    threshold = 0.125
    
    pipeline_NMF(test_data, num_topics, num_top_words, docnames, top_topics_count, threshold)
    
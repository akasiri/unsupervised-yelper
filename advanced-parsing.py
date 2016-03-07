import pickle
import random
import re

def extract_noun_phrases(random_seed, number_of_docs, debug_doc = ""):
    print("Start NP extraction...")
    
    from nltk.chunk.regexp import RegexpParser 
    from nltk import pos_tag    
    
    random.seed(random_seed)
    
    raw_data = pickle.load(open("pickles/list-of-reviews.p", "rb"))
    documents = random.sample(raw_data, number_of_docs)
    
    rpp = RegexpParser(r"""
  NP: {<DT|JJ|NN.*>+}          # Chunk sequences of DT, JJ, NN
  PP: {<IN><NP>}               # Chunk prepositions followed by NP
  VP: {<VB.*><NP|PP|CLAUSE>+$} # Chunk verbs and their arguments
  CLAUSE: {<NP><VP>}           # Chunk NP, VP
  """)

    parsed_doc = ""
    parsed_docs = []

    if len(debug_doc) > 0:
    
        for i in documents:
            tokens = re.sub("(^ )|( $)+", "", re.sub("(\s|\.|\?|,|;|:)+", " ", documents.lower())).split(" ")
            tagged_doc = pos_tag(tokens)
            parsed_docs.append(rpp.parse(tagged_doc))    

    else:
        tokens = re.sub("(^ )|( $)+", "", re.sub("(\s|\.|\?|,|;|:)+", " ", documents.lower())).split(" ")
        tagged_doc = pos_tag(tokens)
        parsed_doc = rpp.parse(tagged_doc)

    
def lemmatize_docs(random_seed, number_of_docs):
    print("Start lemmatization...")
    
    from nltk.stem.wordnet import WordNetLemmatizer
    
    random.seed(random_seed)
    
    raw_data = pickle.load(open("pickles/list-of-reviews.p", "rb"))
    documents = random.sample(raw_data, number_of_docs)

    lemmatized_docs = []
    
    wnl = WordNetLemmatizer()
    for i in documents:
        tokens = re.sub("(^ )|( $)+", "", re.sub("(\s|\.|\?|,|;|:)+", " ", i.lower())).split(" ")
        lemmatized_doc = ""
        for j in tokens:
            lemmatized_doc += wnl.lemmatize(j) + " "
        lemmatized_docs.append(lemmatized_doc)
    
    pickle.dump(lemmatized_docs, open("pickles/lemmatized-docs.p", "wb"))
    print("Lemmatization complete.")
    
    return lemmatized_docs


if __name__ == "__main__":
    lemmatize_docs(1532525625823, 30000)
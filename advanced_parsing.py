import pickle
import random
import re

"""
    Though this will make an easy access complete dtm, you won't have the
    countvect for when you need to build a matrix for the prediction data
"""
def extract_and_store_tfidf_dtm(tokenization_method, review_count):
    # for consistent testing
    random.seed(1532525625823)
    
    raw_data = pickle.load(open("pickles/list-of-reviews.p", "rb"))
    documents = random.sample(raw_data, review_count)    
    
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.feature_extraction.text import TfidfTransformer
    import sklearn.feature_extraction.text as text
            
            
    our_stopwords = "food great good place service like time just really love best nice ve friendly staff photographer ipad trial pole chopstick 2x honeymoon mastro tastefully dramatic tokyo importance ceremony visual checkin wa food great place t s good service time like just really love best nice neighborhood a little pricey cute place rick tequila a bus a bun a combo chinese restaurant the high side restroom great wine"
    stopwords = text.ENGLISH_STOP_WORDS.union(set(our_stopwords.split(" ")))
    
    if tokenization_method == 'np':
        count_vect = CountVectorizer(strip_accents="unicode", tokenizer = extract_np_tokens, stop_words=stopwords)
    elif tokenization_method == 'lm':
        count_vect = CountVectorizer(strip_accents="unicode", tokenizer = extract_lemmatized_tokens, stop_words=stopwords)
    else:
        count_vect = CountVectorizer(strip_accents="unicode", stop_words=stopwords)

    count_matrix = count_vect.fit_transform(documents)

    tfidf_transformer = TfidfTransformer()
    tfidf_matrix = tfidf_transformer.fit_transform(count_matrix)
    
    # so that we don't mave to run this again if we are using 
    pickle.dump(tfidf_matrix, open("pickles/{}-{}-dtm.p".format(tokenization_method, review_count), "wb"))
    pickle.dump(count_vect, open("pickles/{}-{}-count-vect.p".format(tokenization_method, review_count), "wb"))
    pickle.dump(tfidf_transformer, open("pickles/{}-{}-tfidf-trans.p".format(tokenization_method, review_count), "wb"))


def extract_np_tokens( doc: str):
    from nltk.chunk.regexp import RegexpParser 
    from nltk import pos_tag    

    pattern = """NP: {<DT|PP\$>?<JJ>*<NN>}|"""
    rpp = RegexpParser(pattern)

    tokens = re.sub("(^ )|( $)+", "", re.sub("(\W)+", " ", doc.lower())).split(" ")
    tagged_doc = pos_tag(tokens)
    parsed_doc = rpp.parse(tagged_doc)
    
    tokens = []
    
    def traverse(t):
        try:
            t.label
        except AttributeError:
            return
        else:
            if (t.label() == "NP"):
                np = ""
                for (term, pos) in t.leaves():
                    np += term + " "
                tokens.append(np[:-1])            
            
            for child in t:
                traverse(child)

    traverse(parsed_doc)

    return tokens

"""
takes a documents and returns the it's lemmatized tokens
"""
def extract_lemmatized_tokens(doc):
    from nltk.stem.wordnet import WordNetLemmatizer

    wnl = WordNetLemmatizer()
    lemmatized_tokens = []

    tokens = re.sub("(^ )|( $)+", "", re.sub("(\W)+", " ", doc.lower())).split(" ")
    for j in tokens:
        lemmatized_tokens.append(wnl.lemmatize(j))
    
    return lemmatized_tokens


if __name__ == "__main__":
    # EXAMPLE / TEST RUNS
#    doc = """The Union was formed in September 1921 by the merger of three left-wing trade unions that had not joined the Allgemeiner Deutscher Gewerkschaftsbund (ADGB), which they, like other radicalized workers in the General Workers Union of Germany (Allgemeine Arbeiter-Union Deutschlands) and the Free Workers' Union of Germany had felt was reformist. The three unions were the Gelsenkirchen Free Workers' Union, the Berlin-based Association of Manual and Intellectual Workers and the Braunschweig-based Farmworkers' Association (Landarbeiterverband). Gustav Sobottka was one of the founding members of the union. At the national level, the newly merged Union became part of the Profintern. The Union's was mainly focused in the Ruhr region and bordering areas, as well as in the Berlin area. The dominant sectors were mining and metalworking. In the Ruhr region, about half the KPD members who were members of various trade unions were also members of the Union."""
#    print(extract_np_tokens(doc))
#    print(extract_lemmatized_tokens(doc))
    
    extract_and_store_tfidf_dtm('np', 30000)
    extract_and_store_tfidf_dtm('lm', 30000)
    extract_and_store_tfidf_dtm('default', 30000)
    
    
    
    
    
    
    
    
    
    
    
    
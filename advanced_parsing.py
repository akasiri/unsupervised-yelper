import pickle
import random
import re

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


def extract_np_tokens( doc: str):
    from nltk.chunk.regexp import RegexpParser 
    from nltk import pos_tag    

    pattern = """NP: {<DT|PP\$>?<JJ>*<NN>}|"""
    rpp = RegexpParser(pattern)

    tokens = re.sub("(^ )|( $)+", "", re.sub("(\s|\.|\?|,|;|:|\(|\))+", " ", doc.lower())).split(" ")
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
                tokens.append(np)            
            
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

    tokens = re.sub("(^ )|( $)+", "", re.sub("(\s|\.|\?|,|;|:)+", " ", doc.lower())).split(" ")
    for j in tokens:
        lemmatized_tokens.append(wnl.lemmatize(j))
    
    return lemmatized_tokens


if __name__ == "__main__":
    #lemmatize_docs(1532525625823, 30000)
    doc = """The Union was formed in September 1921 by the merger of three left-wing trade unions that had not joined the Allgemeiner Deutscher Gewerkschaftsbund (ADGB), which they, like other radicalized workers in the General Workers Union of Germany (Allgemeine Arbeiter-Union Deutschlands) and the Free Workers' Union of Germany had felt was reformist. The three unions were the Gelsenkirchen Free Workers' Union, the Berlin-based Association of Manual and Intellectual Workers and the Braunschweig-based Farmworkers' Association (Landarbeiterverband). Gustav Sobottka was one of the founding members of the union. At the national level, the newly merged Union became part of the Profintern. The Union's was mainly focused in the Ruhr region and bordering areas, as well as in the Berlin area. The dominant sectors were mining and metalworking. In the Ruhr region, about half the KPD members who were members of various trade unions were also members of the Union."""
    print(extract_np_tokens(doc))
    print(extract_lemmatized_tokens(doc))
    
    
    
    
    
    
    
    
    
    
    
import pickle

def build_topic_file(name):
    doctopic, topic_words = pickle.load(open("pickles/"+name+".p", "rb"))
    with open(name+".txt", "w") as f:
        for topic in topic_words:
            for word in topic:
                f.write(word.replace(" ", "_") + " ")
            f.write("\n")
        

def build_intruder_files():
    pass

if __name__ == '__main__':
    build_topic_file("wo-stop-words_nmf-np-60-doctopic-topic_words")
    build_intruder_files()
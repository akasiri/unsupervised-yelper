import json
import pickle
from collections import defaultdict

# Am aware that there is the risk of two restaurants with the same name, which 
# in our code will map to the same restraunt, but if that is the case we are in
# a bind either way. We would not be able to discern which restaurant the user 
# was looking for, based solely on the name given to us.

reviews = []
business_names = []
busiid_to_businame = dict()
busi_to_rev = defaultdict(list)

with open("yelp_dataset_challenge_academic_dataset\yelp_academic_dataset_business.json") as fp:    
    for line in fp:
        business = json.loads(line)
        business_names.append(business["name"])
        busiid_to_businame[business["business_id"]] = business["name"]

with open("yelp_dataset_challenge_academic_dataset\yelp_academic_dataset_review.json") as fp:    
    for line in fp:
        review = json.loads(line)
        
        reviews.append(review["text"])
        busi_to_rev[busiid_to_businame[review["business_id"]]].append(review["text"])

pickle.dump(reviews, open("pickles/list-of-reviews.p", "wb"))
pickle.dump(business_names, open("pickles/list-of-business-names.p", "wb"))
pickle.dump(busi_to_rev, open("pickles/dict-of-business-to-reviews.p", "wb"))

#training_data = []
#
## LEMMATIZING THE TOKENS
#print("Start lemmatization...")
#wnl = WordNetLemmatizer()
#for i in training_documents:
#    tokens = re.sub("(^ )|( $)+", "", re.sub("(\s|\.|\?|,|;|:)+", " ", i.lower())).split(" ")
#    lemmatized_doc = ""
#    for j in tokens:
#        lemmatized_doc += wnl.lemmatize(j) + " "
#    training_data.append(lemmatized_doc)
#print("Lemmatization complete.")
#
#pickle.dump(training_data, open("pickles/lemmatized-docs.p", "wb"))
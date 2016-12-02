# Import dataset 
from sklearn.datasets import fetch_20newsgroups

# Refactoring categories into the broad topics
science_categories = ['sci.crypt', 'sci.electronics', 'sci.med', 'sci.space'] #0
religion_categories = ['talk.religion.misc','alt.atheism','soc.religion.christian']  #1
politics_categories = [ 'talk.politics.misc', 'talk.politics.guns','talk.politics.mideast' ] #2
misc_categories = ['misc.forsale'] #3
sports_categories = ['rec.autos', 'rec.motorcycles', 'rec.sport.baseball', 'rec.sport.hockey'] #4
computer_categories = [ 'comp.graphics', 'comp.os.ms-windows.misc', 'comp.sys.ibm.pc.hardware', 'comp.sys.mac.hardware', 'comp.windows.x' ] #5

final_categories = ['science', 'religion', 'politics', 'misc', 'sports', 'computer']

all_categories = science_categories + religion_categories + politics_categories + misc_categories + sports_categories + computer_categories

newsgroups_train_original = fetch_20newsgroups(subset='train', categories=all_categories,remove=('footers'))

import re

#Regex expression to remove email addresses
EMAIL_REGEX = re.compile(r"([a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+)")

convert_target_hash = {}
convert_target_hash[0] = 1
convert_target_hash[1] = 5
convert_target_hash[2] = 5
convert_target_hash[3] = 5
convert_target_hash[4] = 5
convert_target_hash[5] = 5
convert_target_hash[6] = 3
convert_target_hash[7] = 4
convert_target_hash[8] = 4
convert_target_hash[9] = 4
convert_target_hash[10] = 4
convert_target_hash[11] = 0
convert_target_hash[12] = 0
convert_target_hash[13] = 0
convert_target_hash[14] = 0
convert_target_hash[15] = 1
convert_target_hash[16] = 2
convert_target_hash[17] = 2
convert_target_hash[18] = 2
convert_target_hash[19] = 1

for i in range(0,len(newsgroups_train_original.data)):
    split_words = newsgroups_train_original.data[i].split('\n')
    newsgroups_train_original.data[i] = ""
    newsgroups_train_original.target[i] = convert_target_hash[newsgroups_train_original.target[i]]
    for j in range(0,len(split_words)):
        if EMAIL_REGEX.search(split_words[j]) is None:
            newsgroups_train_original.data[i] += split_words[j] +'\n'

import nltk
from nltk.corpus import stopwords
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
import nltk.data, nltk.tag
from nltk.tag.perceptron import PerceptronTagger
import numpy as np
import string

# To lemmatize tokens
class LemmaTokenizer(object):
    def __init__(self):
        self.wnl = WordNetLemmatizer()
        self.tagger = PerceptronTagger()
    def __call__(self, doc):
        self.wnl.decode_error = "ignore"
        helper = [self.wnl.lemmatize(t) for t in word_tokenize(doc)]
        helper = [i for i in helper if i not in string.punctuation]
        tagset = None
        tagged = nltk.tag._pos_tag(helper,tagset,self.tagger)
        nouns = [word for word,pos in tagged if pos == 'NN'or pos == 'NNP']
        return nouns

# Stopwords extracted from text as analysed by count vectoriser
other_stopwords = ['article','bit','book','chip','day','doe','ha','hand','line','lot','number','organization',
                   'person','place','point','problem','question','time','university','version','wa','way',
                   'week','work','world','year','u']
stopwords_set = set(stopwords.words('english'))
stopwords_set.update(other_stopwords)

for i in range(0,len(newsgroups_train_original.data)):
    split_words = word_tokenize(newsgroups_train_original.data[i])
    newsgroups_train_original.data[i] = ""
    for j in range(0, len(split_words)):
        if split_words[j].lower() not in stopwords_set:
            newsgroups_train_original.data[i] += split_words[j] + ' '

count_vect = CountVectorizer(tokenizer=LemmaTokenizer(), decode_error="ignore", stop_words=stopwords_set, max_features=100)

# To transform words into features and correpsonding counts
train_features = count_vect.fit_transform(newsgroups_train_original.data)
train_features = train_features.toarray()
vocab = count_vect.get_feature_names()
for i in range (0,40):
	print vocab[i]

dist = np.sum(train_features, axis=0)

# Newsgroups test dataset
newsgroups_test = fetch_20newsgroups(subset='test', remove=('footers'), categories=all_categories)

for i in range(0,len(newsgroups_test.data)):
    split_words = newsgroups_test.data[i].split('\n')
    newsgroups_test.data[i] = ""
    newsgroups_test.target[i] = convert_target_hash[newsgroups_test.target[i]]
    for j in range(0,len(split_words)):
        if EMAIL_REGEX.search(split_words[j]) is None:
            newsgroups_test.data[i] += split_words[j] +'\n'

for i in range(0,len(newsgroups_test.data)):
    split_words = word_tokenize(newsgroups_test.data[i])
    newsgroups_test.data[i] = ""
    for j in range(0, len(split_words)):
        if split_words[j].lower() not in stopwords_set:
            newsgroups_test.data[i] += split_words[j] + ' '

y_test = newsgroups_test.target
y_train = newsgroups_train_original.target

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer

# Tf-IDF Vectoriser to generate features set 
vectorizer = TfidfVectorizer(tokenizer=LemmaTokenizer(), stop_words='english', use_idf=True)
vectors = vectorizer.fit_transform(newsgroups_train_original.data)
vectors_test = vectorizer.transform(newsgroups_test.data)

# Dimensionality reduction using SVD for effective KNN run
svd = TruncatedSVD(1000)
lsa = make_pipeline(svd, Normalizer(copy=False))
vectors = lsa.fit_transform(vectors)

explained_variance = svd.explained_variance_ratio_.sum()
print("  Explained variance of the SVD step: {}%".format(int(explained_variance * 100)))

vectors_test = lsa.transform(vectors_test)

from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier

# K- Nearest Neighbors
knn = KNeighborsClassifier(n_neighbors=10, algorithm='brute', p=2)
knn.fit(vectors, newsgroups_train_original.target)
knn_pred_train = knn.predict(vectors)
knn_pred_test = knn.predict(vectors_test)
knn_f1 = metrics.f1_score(newsgroups_test.target, knn_pred_test, average='macro')

# Multinomial Naive Bayes
mnb = MultinomialNB(alpha=.01)
mnb.fit(vectors, newsgroups_train_original.target)
mnb_pred_train = mnb.predict(vectors)
mnb_pred_test = mnb.predict(vectors_test)
mnb_f1 = metrics.f1_score(newsgroups_test.target, mnb_pred, average='macro')

rfc = RandomForestClassifier(n_estimators=300)
rfc.fit(vectors,  newsgroups_train_original.target)
rfc_pred_train = rfc.predict(vectors)
rfc_pred_test = rfc.predict(vectors_test)
rfc_f1 = metrics.f1_score(newsgroups_test.target, rfc_pred, average='macro')

svm = SGDClassifier(loss='hinge', penalty='l2', alpha=1e-3, n_iter=5, random_state=42)
svm.fit(vectors,  newsgroups_train_original.target)
svm_pred_train = svm.predict(vectors)
svm_pred_test = svm.predict(vectors_test)
svm_f1 = metrics.f1_score(newsgroups_test.target, svm_pred, average='macro')

from sknn.mlp import Classifier, Layer
from sklearn.model_selection import train_test_split

pred_1_train = np.vstack((knn_pred_train, mnb_pred_train, rfc_pred_train, svm_pred_train))
pred_1_test = np.vstack((knn_pred_test, mnb_pred_test, rfc_pred_test, svm_pred_test))

pred_1_train = pred_1_train.T
pred_1_test = pred_1_test.T

# Feeding into MLP
nn = Classifier(
layers=[
Layer("Sigmoid", units=100),
Layer("Softmax")],
learning_rate=0.001,
n_iter=25)

newsgroups_train_original.target = newsgroups_train_original.target.reshape(11314,1)
nn.fit(pred_1_train, newsgroups_train_original.target)

y_example = nn.predict(pred_1_test)

y_test.shape
u = nn.score(y_test, y_example)

# Accuracy of MLP
from sklearn.metrics import accuracy_score
print (accuracy_score(y_example, y_test))

target_names=[0,1,2,3,4,5]
newsgroups_test.target = newsgroups_test.target.reshape(7532,1)
classification_report(newsgroups_test.target, mnb_pred_test, target_names)
print classification_report

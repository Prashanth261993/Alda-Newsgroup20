from sklearn.datasets import fetch_20newsgroups

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

nltk.download('wordnet')

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

dist = np.sum(train_features, axis=0)
for tag, count in zip(vocab, dist):
    print (count+',', tag)


#print filter(lambda w: not w in stopwords_set,newsgroups_train.data[1].split())

"""
is_noun = lambda pos: pos[:2] == 'NN'
# do the nlp stuff
tokenized = nltk.word_tokenize(lines)
nouns = [word for (word, pos) in nltk.pos_tag(tokenized) if is_noun(pos)]

print nouns
>>> ['lines', 'string', 'words']
"""

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


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics

vectorizer = TfidfVectorizer(tokenizer=LemmaTokenizer())
vectors = vectorizer.fit_transform(newsgroups_train_original.data)
vectors_test = vectorizer.transform(newsgroups_test.data)
multinomial_clf = MultinomialNB(alpha=.01)
multinomial_clf.fit(vectors, newsgroups_train_original.target)
pred = multinomial_clf.predict(vectors_test)
metrics.f1_score(newsgroups_test.target, pred, average='macro')

def show_top20(classifier, vectorizer, categories):
    feature_names = np.asarray(vectorizer.get_feature_names())
    for i, category in enumerate(categories):
        top20 = np.argsort(classifier.coefs_[i])[-10:]
        print("%s: %s" % (category, " ".join(feature_names[top20])))

# further considerations for stopwords: could, like, subject, lines, would, nntp, posting, edu, one

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(newsgroups_test.target, pred)

norm_conf = []
for i in cm:
    a = 0
    tmp_arr = []
    a = sum(i, 0)
    for j in i:
        tmp_arr.append(float(j)/float(a))
    norm_conf.append(tmp_arr)

fig = plt.figure()
plt.clf()
ax = fig.add_subplot(111)
ax.set_aspect(1)
res = ax.imshow(np.array(norm_conf), cmap=plt.cm.jet,
                interpolation='nearest')

width, height = cm.shape

for x in xrange(width):
    for y in xrange(height):
        ax.annotate(str(cm[x][y]), xy=(y, x),
                    horizontalalignment='center',
                    verticalalignment='center')

cb = fig.colorbar(res)

plt.xticks(range(width), final_categories[:width])
plt.yticks(range(height), final_categories[:height])
plt.savefig('confusion_matrix.png', format='png')

import seaborn as sn
import pandas as pd
df_cm = pd.DataFrame(cm, index = [i for i in final_categories],
                  columns = [i for i in final_categories])
plt.figure(figsize = (10,7))
sn.heatmap(df_cm, annot=True, fmt='g')

from sklearn.neural_network import MLPClassifier
clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(15,), random_state=1, max_iter= 100000)
clf.fit(vectors, newsgroups_train_original.target)

from sklearn.ensemble import RandomForestClassifier
forest = RandomForestClassifier(n_estimators = 100)

# Fitting the forest may take a few minutes
print "Fitting a random forest to labeled training data..."
forest = forest.fit(vectors,newsgroups_train_original.target)
result = forest.predict(vectors_test)

from sklearn.linear_model import SGDClassifier
text_clf = Pipeline([('vect', CountVectorizer()),
                      ('tfidf', TfidfTransformer()),
                      ('clf', SGDClassifier(loss='hinge', penalty='l2',
                                            alpha=1e-3, n_iter=5, random_state=42)),
 ])
svm_clf = SGDClassifier(loss='hinge', penalty='l2', alpha=1e-3, n_iter=5, random_state=42)
svm_clf.fit(vectors, newsgroups_train_original.target)
predicted = svm_clf.predict(vectors_test)


# Combining multiple classifiers

multinomial_clf.fit(vectors, newsgroups_train_original.target)
pred_train = multinomial_clf.predict(vectors)

knn_train.fit(vectors, newsgroups_train_original.target)
predicted_values_train = svm_clf.predict(vectors)

svm_clf.fit(vectors, newsgroups_train_original.target)
predicted_train = svm_clf.predict(vectors)



y_train = [[[] for i in range(5)] for i in range(11314)]

for x in range(0,11314):
    y_train[x][0] = pred_train[x]
    y_train[x][1] = svm_train[x]
    y_train[x][2] = knn_train[x]
    y_train[x][3] = rtree_train[x]
    y_train[x][4] = mlp_train[x]

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(y_train)
y_train = scaler.transform(y_train)

clf = MLPClassifier(solver='adam', alpha=1e-5, hidden_layer_sizes=(5,2), random_state=42, max_iter= 100000)
clf.fit(y_train, newsgroups_train_original.target)

predicted = svm_clf.predict(vectors_test)

pred = multinomial_clf.predict(vectors_test)

predicted_values = svm_clf.predict(vectors_test)

y_test = [[[] for i in range(5)] for i in range(7532)]

from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=10, algorithm='brute', p=2)
knn.fit(vectors, newsgroups_train_original.target)
knn_pred_train = knn.predict(vectors)
knn_pred_test = knn.predict(vectors_test)
knn_f1 = metrics.f1_score(newsgroups_test.target, knn_pred_test, average='macro')

for x in range(0,7532):
    y_test[x][0] = pred_test[x]
    y_test[x][1] = svm_test[x]
    y_test[x][2] = knn_test[x]
    y_test[x][3] = rtree_test[x]
    y_test[x][4] = mlp_test[x]

y_test = scaler.transform(y_test)
final_prediction = clf.predict(y_test)
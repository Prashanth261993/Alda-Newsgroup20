# -*- coding: utf-8 -*-
"""
Created on Mon Nov 07 16:54:11 2016

@author: swati
"""
from gensim.models import Word2Vec
from sklearn.pipeline import Pipeline
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.cluster import KMeans
from sklearn.datasets import load_files
from sklearn import tree
from nltk import word_tokenize 
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import *
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier, Perceptron
from sklearn.metrics import confusion_matrix
from nltk.corpus import stopwords
from sklearn.metrics import classification_report
import numpy as np
import gensim
import re
import nltk
import matplotlib.pyplot as plt
import itertools
tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')
model = Word2Vec.load("word2vec_with_six_categories")
import time
trial = load_files('20news-bydate-train')
test  = load_files('20news-bydate-test')
articles_final = []
science_categories = ['sci.crypt', 'sci.electronics', 'sci.med', 'sci.space'] #0 
religion_categories = ['talk.religion.misc','alt.atheism','soc.religion.christian']  #1 
politics_categories = [ 'talk.politics.misc', 'talk.politics.guns','talk.politics.mideast' ] #2 
misc_categories = ['misc.forsale'] #3 
sports_categories = ['rec.autos', 'rec.motorcycles', 'rec.sport.baseball', 'rec.sport.hockey'] #4 
computer_categories = [ 'comp.graphics', 'comp.os.ms-windows.misc', 'comp.sys.ibm.pc.hardware', 'comp.sys.mac.hardware', 'comp.windows.x' ] #5 
 
final_categories = ['science', 'religion', 'politics', 'misc', 'sports', 'computer'] 
 
all_categories = science_categories + religion_categories + politics_categories + misc_categories + sports_categories + computer_categories 
 
#newsgroups_train_original = load_files('20news-bydate-train')
 
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
 
for i in range(0,len(trial.data)): 
    split_words = trial.data[i].split('\n') 
    trial.data[i] = "" 
    trial.target[i] = convert_target_hash[trial.target[i]] 
    for j in range(0,len(split_words)): 
        if EMAIL_REGEX.search(split_words[j]) is None: 
            trial.data[i] += split_words[j] +'\n' 
for i in range(0,len(test.data)): 
    split_words = test.data[i].split('\n') 
    test.data[i] = "" 
    test.target[i] = convert_target_hash[test.target[i]] 
    for j in range(0,len(split_words)): 
        if EMAIL_REGEX.search(split_words[j]) is None: 
            test.data[i] += split_words[j] +'\n' 
 
 
start = time.time() # Start time


word_vectors = model.syn0
num_clusters = 12

kmeans_clustering = KMeans( n_clusters = num_clusters )
idx = kmeans_clustering.fit_predict( word_vectors )


end = time.time()
elapsed = end - start
print "Time taken for K Means clustering: ", elapsed, "seconds."
                                                                                           
word_centroid_map = dict(zip( model.index2word, idx ))
# For the first 10 clusters

for cluster in xrange(0,10):
    #
    # Print the cluster number  
    print "\nCluster %d" % cluster
    #
    # Find all of the words for that cluster number, and print them out
    words = []
    for i in xrange(0,len(word_centroid_map.values())):
        if( word_centroid_map.values()[i] == cluster ):
            words.append(word_centroid_map.keys()[i])
    print words


def sentence_to_words_helper(sent):
     sent = str(sent)
     sent = re.sub("[^a-zA-Z]"," ", sent)
     words = sent.lower().split()
     stop_words = set(stopwords.words("english"))
     words = [w for w in words if not w in stop_words]
     return words
       
#Splits paragraphs into array of sentence
#def sentence_wordlist(data_articles):
final_input_word_list = []
for i in range(len(trial.data)):
    # per document
   # list_of_sentences = tokenizer.tokenize(trial.data[i])
    #list_of_sentences = re.split(r'(?<=[^A-Z].[.?]) +(?=[A-Z])', trial.data[i])
        #print("\n".join(doc.split('\n')))
        #print type(list_of_sentences)
   ## sentences_words_list = []
    #for sentence in list_of_sentences:
        #one sent of the doc
        #sentences_words_list.append(sentence_to_words_helper(sentence))
    final_input_word_list.append(sentence_to_words_helper(trial.data[i]))
print "length of final_input_wordlist"
print len(final_input_word_list)

final_test_word_list = []
for i in range(len(test.data)):
    # per document
    #list_of_sentences = tokenizer.tokenize(test.data[i])
    #list_of_sentences = re.split(r'(?<=[^A-Z].[.?]) +(?=[A-Z])', trial.data[i])
        #print("\n".join(doc.split('\n')))
        #print type(list_of_sentences)
    #sentences_test_list = []
    #for sentence in list_of_sentences:
        #one sent of the doc
        #sentences_test_list.append(sentence_to_words_helper(sentence))
    final_test_word_list.append(sentence_to_words_helper(test.data[i]))

#sentences_words_list.append(sentence_to_words_helper(list_of_sentences))
#print sentences_words_list...
#print len(final_input_word_list)

def create_bag_of_centroids( wordlist, word_centroid_map ):

    num_centroids = max( word_centroid_map.values() ) + 1
 
    bag_of_centroids = np.zeros( num_centroids, dtype="float32" )
  
    for word in wordlist:
        if word in word_centroid_map:
            index = word_centroid_map[word]
            bag_of_centroids[index] += 1
    #
    # Return the "bag of centroids"
    return bag_of_centroids
    
# Pre-allocate an array for the training set bags of centroids (for speed)
train_centroids = np.zeros( (len(trial.data), num_clusters), \
    dtype="float32" )

# Transform the training set reviews into bags of centroids
counter = 0
#train_wordlist = sentence_wordlist(trial.data)

for word_list in final_input_word_list:
    train_centroids[counter] = create_bag_of_centroids( word_list, \
        word_centroid_map )
    counter += 1

# Repeat for test reviews 
test_centroids = np.zeros(( len(test.data), num_clusters), \
    dtype="float32" )

counter = 0
#test_wordlist = sentence_wordlist(test.data)
for word_list in final_test_word_list:
    test_centroids[counter] = create_bag_of_centroids( word_list, \
        word_centroid_map )
    counter += 1

# Fit a random forest and extract predictions 
forest = RandomForestClassifier(n_estimators = 100)

# Fitting the forest may take a few minutes
print "Fitting a random forest to labeled training data..."
forest = forest.fit(train_centroids,trial.target)
result = forest.predict(test_centroids)
print result[1]

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

print "Random forest classifier"
# Compute confusion matrix
cnf_matrix = confusion_matrix(result, test.target)
np.set_printoptions(precision=2)

# Plot non-normalized confusion matrix
#plt.figure()
plot_confusion_matrix(cnf_matrix, classes=trial.target,
                      title='Confusion matrix, without normalization')

# Plot normalized confusion matrix
#plt.figure()
plot_confusion_matrix(cnf_matrix, classes=test.target, normalize=True,
                      title='Normalized confusion matrix')

#plt.show()
#plt.savefig("out.png", transparent = True)
print(classification_report(test.target,result))


print "SVM"
from sklearn import svm
clf = svm.SVC()
clf.fit(train_centroids, trial.target)
clf.predict(test_centroids)
result_svm = clf.predict(test_centroids)
cnf_matrix_svm = confusion_matrix(result_svm, test.target)
plot_confusion_matrix(cnf_matrix_svm, classes=trial.target,
                      title='Confusion matrix, without normalization')
plot_confusion_matrix(cnf_matrix_svm, classes=test.target, normalize=True,
                      title='Normalized confusion matrix')
print(classification_report(test.target,result_svm))
print("NB")

from sklearn.naive_bayes import MultinomialNB
clf_NB = MultinomialNB(alpha=.01) 
clf_NB.fit(train_centroids, trial.target) 
result_NB = clf_NB.predict(test_centroids)
cnf_matrix_nb = confusion_matrix(result_NB, test.target)
plot_confusion_matrix(cnf_matrix_nb, classes=trial.target,
                      title='Confusion matrix, without normalization')
print(classification_report(test.target,result_NB))     


print("Check for Perceptron")
clf_perceptron = Perceptron()
clf_perceptron.fit(train_centroids, trial.target)
result_per = clf_perceptron.predict(test_centroids)
cnf_matrix_per = confusion_matrix(result_per, test.target)
plot_confusion_matrix(cnf_matrix_per, classes=trial.target,
                      title='Confusion matrix, without normalization')
print(classification_report(test.target,result_per))  

                      
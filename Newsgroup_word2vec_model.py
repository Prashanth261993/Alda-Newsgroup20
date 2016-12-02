# -*- coding: utf-8 -*-
"""
Created on Mon Nov 21 16:47:51 2016

@author: swati
"""

#from sklearn.datasets import fetch_20newsgroups 
import gensim
from gensim.models import word2vec
from sklearn.datasets import load_files
science_categories = ['sci.crypt', 'sci.electronics', 'sci.med', 'sci.space'] #0 
religion_categories = ['talk.religion.misc','alt.atheism','soc.religion.christian']  #1 
politics_categories = [ 'talk.politics.misc', 'talk.politics.guns','talk.politics.mideast' ] #2 
misc_categories = ['misc.forsale'] #3 
sports_categories = ['rec.autos', 'rec.motorcycles', 'rec.sport.baseball', 'rec.sport.hockey'] #4 
computer_categories = [ 'comp.graphics', 'comp.os.ms-windows.misc', 'comp.sys.ibm.pc.hardware', 'comp.sys.mac.hardware', 'comp.windows.x' ] #5 
 
final_categories = ['science', 'religion', 'politics', 'misc', 'sports', 'computer'] 
 
all_categories = science_categories + religion_categories + politics_categories + misc_categories + sports_categories + computer_categories 
 
newsgroups_train_original = load_files('20news-bydate-train')
 
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
from nltk import word_tokenize 
from nltk.stem import WordNetLemmatizer 
from sklearn.feature_extraction.text import CountVectorizer 
import nltk.data, nltk.tag 
from nltk.tag.perceptron import PerceptronTagger 
#nltk.download()
import numpy as np 
import string 
from nltk.corpus import stopwords
tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
tagger = PerceptronTagger()
wnl = WordNetLemmatizer()
tagset = None 
nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
 
 
other_stopwords = ['article','bit','book','chip','day','doe','ha','hand','line','lot','number','organization', 
                   'person','place','point','problem','question','time','university','version','wa','way', 
                   'week','work','world','year','u'] 
stop_words = set(stopwords.words('english')) 
stop_words.update(other_stopwords) 

            
def sentence_to_words_helper(sent):
     sent = str(sent)
     sent = re.sub("[^a-zA-Z]"," ", sent)
     words = sent.lower().split()
     words = [w for w in words if not w in stop_words]
     words = [wnl.lemmatize(t) for t in words]
     tagged = nltk.tag._pos_tag(words,tagset,tagger)
     nouns = [word for word,pos in tagged if pos == 'NN'or pos == 'NNP']
     return nouns

       
#Splits paragraphs into array of sentence
final_input_word_list = []
for i in range(len(newsgroups_train_original.data)):
    #list_of_sentences = tokenizer.tokenize(trial.data[i])
    list_of_sentences = re.split(r'(?<=[^A-Z].[.?]) +(?=[A-Z])', newsgroups_train_original.data[i])
    #print("\n".join(doc.split('\n')))
    #print type(list_of_sentences)
    sentences_words_list = []
    for sentence in list_of_sentences:
        sentences_words_list.append(sentence_to_words_helper(sentence))
        final_input_word_list +=sentences_words_list
#sentences_words_list.append(sentence_to_words_helper(list_of_sentences))
#print sentences_words_list...
print len(final_input_word_list)

num_features = 400    # Word vector dimensionality                      
min_word_count = 10   # Minimum word count                        
num_workers = 4       # Number of threads to run in parallel
context = 10          # Context window size                                                                                    
downsampling = 1e-3   # Downsample setting for frequent words
print final_input_word_list[1]
model = word2vec.Word2Vec(final_input_word_list, workers=num_workers, \
            size=num_features, min_count = min_word_count, \
            window = context, sample = downsampling)
model.init_sims(replace=True)
model_name = "word2vec_with_six_categories_400_dim"
model.save(model_name)

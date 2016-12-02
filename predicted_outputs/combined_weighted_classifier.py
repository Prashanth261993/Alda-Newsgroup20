import numpy as np
from sklearn.neural_network import MLPClassifier
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
newsgroups_test = fetch_20newsgroups(subset='test', remove=('footers'), categories=all_categories)

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

# Mapping the target labels of training data from 20 to 6 categories

for i in range(0,len(newsgroups_train_original.data)):
    newsgroups_train_original.target[i] = convert_target_hash[newsgroups_train_original.target[i]]

for i in range(0,len(newsgroups_test.data)):
    newsgroups_test.target[i] = convert_target_hash[newsgroups_test.target[i]]

# Combining multiple classifiers


#Load predictions from different models
mnb_train = np.loadtxt('mnb_pred_train')
mnb_test = np.loadtxt('mnb_pred_test')
svm_train = np.loadtxt('svm_train_pred.txt')
svm_test = np.loadtxt('svm_test_pred.txt')
knn_train = np.loadtxt('knn-train.txt')
knn_test = np.loadtxt('knn-test.txt')
rtree_train = np.loadtxt('rtree_train_word2vec')
rtree_test = np.loadtxt('rtree_test_word2vec')
mlp_train = np.loadtxt('mlp_train_predict')
mlp_test = np.loadtxt('mlp_train_predict')


y_train = [[[] for i in range(5)] for i in range(11314)]

for x in range(0,11314):
    y_train[x][0] = mnb_train[x]
    y_train[x][1] = svm_train[x]
    y_train[x][2] = knn_train[x]
    y_train[x][3] = rtree_train[x]
    y_train[x][4] = mlp_train[x]

#Scale the data, MLP performs much better after scaling, Range: [-1,1]
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(y_train)
y_train = scaler.transform(y_train)

clf = MLPClassifier(solver='adam', alpha=1e-5, hidden_layer_sizes=(5,2), random_state=42, max_iter= 100000)
clf.fit(y_train, newsgroups_train_original.target)

y_test = [[[] for i in range(5)] for i in range(7532)]

for x in range(0,7532):
    y_test[x][0] = mnb_test[x]
    y_test[x][1] = svm_test[x]
    y_test[x][2] = knn_test[x]
    y_test[x][3] = rtree_test[x]
    y_test[x][4] = mlp_test[x]

y_test = scaler.transform(y_test)
final_prediction = clf.predict(y_test)

from sklearn import metrics
metrics.f1_score(newsgroups_test.target, final_prediction, average='macro')

# Confusion Matrix of actual vs predicted newsgroup articles based on MNB

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(newsgroups_test.target, final_prediction)

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
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score,roc_curve, auc
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Dropout, Embedding, LSTM, Bidirectional
from nltk import word_tokenize, pos_tag
from nltk.corpus import wordnet
from nltk.stem.porter import PorterStemmer

stopWords = set(stopwords.words('english'))
stopWords.add('``')
stopWords.add('<')
stopWords.add('/')
stopWords.add('>')
stopWords.add('.')
stopWords.add('!')
stopWords.add("''")
stopWords.add('-')
stopWords.add('(')
stopWords.add(')')

neg_path = "rt-polarity.neg"
pos_path = "rt-polarity.pos"

comments = []
labels = []
        
with open(neg_path,"r",encoding="utf-8",errors="ignore") as r:
    for line in r:
        comments.append(line)
        labels.append('neg')
        
with open(pos_path,"r",encoding="utf-8",errors="ignore") as r:
    for line in r:
        comments.append(line)
        labels.append('pos')
        
####remove stop words 
#comments_stpwords = []
#for comment in comments:
#    splitted = comment.strip().split(' ')
#    words = []
#    for word in splitted:
#        if word not in stopWords:
#            words.append(word)
#    comment_new = (' ').join(words)
#    comments_stpwords.append(comment_new)
#comments = comments_stpwords
####

####lemmatize
#def get_wordnet_pos(tag):
#    if tag.startswith('J'):
#        return wordnet.ADJ
#    elif tag.startswith('V'):
#        return wordnet.VERB
#    elif tag.startswith('N'):
#        return wordnet.NOUN
#    elif tag.startswith('R'):
#        return wordnet.ADV
#    else:
#        return None
#wlm = WordNetLemmatizer()
#comments_wlm = []
#for comment in comments:
#    tokens = word_tokenize(comment)
#    tagged_sent = pos_tag(tokens)
#    lemmas_sent = []
#    for tag in tagged_sent:
#        wordnet_pos = get_wordnet_pos(tag[1]) or wordnet.NOUN
#        lemmas_sent.append(wlm.lemmatize(tag[0], pos=wordnet_pos))
#    comment_new = (' ').join(lemmas_sent)
#    comments_wlm.append(comment_new)
#comments = comments_wlm
####

####stem
pstm = PorterStemmer()
comments_stm = []
for comment in comments:
    splitted = comment.strip().split(' ')
    words = []
    for word in splitted:
        word = pstm.stem(word)
        words.append(word)
    comment_new = (' ').join(words)
    comments_stm.append(comment_new)
comments = comments_stm
####
        
#####1st time Vectorizer
#vectorizer =  CountVectorizer(max_features = None, min_df = 1, 
#                              max_df = 1.0 , ngram_range = (1,1), stop_words = None, 
#                              analyzer = 'word', lowercase = True)
#####

#####2nd time Vectorizer
#vectorizer =  CountVectorizer(max_features = 2500, min_df = 1, 
#                              max_df = 1.0 , ngram_range = (1,1), stop_words = None, 
#                              analyzer = 'word', lowercase = True)
#####

#####3rd time Vectorizer
#vectorizer =  CountVectorizer(max_features = 2500, min_df = 10, 
#                              max_df = 1.0 , ngram_range = (1,1), stop_words = None, 
#                              analyzer = 'word', lowercase = True)
#####

#####4th time Vectorizer
vectorizer =  CountVectorizer(max_features = 2500, min_df = 10, 
                              max_df = 0.8 , ngram_range = (1,1), stop_words = None, 
                              analyzer = 'word', lowercase = True)
#####

#####5th time Vectorizer
#vectorizer =  CountVectorizer(max_features = 1500, min_df = 10, 
#                              max_df = 0.8 , ngram_range = (1,1), stop_words = None, 
#                              analyzer = 'word', lowercase = True)
#####

#####6th time Vectorizer
#vectorizer =  CountVectorizer(max_features = 1000, min_df = 10, 
#                              max_df = 0.5 , ngram_range = (1,1), stop_words = None, 
#                              analyzer = 'word', lowercase = True)
#####

processed_features = vectorizer.fit_transform(comments).toarray()
X_train, X_test, y_train, y_test = train_test_split(processed_features, labels, test_size=0.2, random_state=0)

####Logistic Regression
#log_model = LogisticRegression(max_iter=200)
#log_model = log_model.fit(X=X_train, y=y_train)
#predictions = log_model.predict(X_test)
#y_score = log_model.decision_function(X_test)
####

####Naive Bayes
#nb = GaussianNB()
#nb.fit(X_train, y_train)
#predictions = nb.predict(X_test)
#y_score = nb.predict_proba(X_test)
#y_score = y_score[:,1]
####

####SVM
#svm = SVC(C=1.0, kernel = 'linear', shrinking=True, 
#                probability=False, class_weight=None, 
#                max_iter=-1, decision_function_shape='ovr', 
#                random_state=None)
#svm.fit(X_train, y_train)
#predictions = svm.predict(X_test)
#y_score = svm.decision_function(X_test)
####

####RandomForest
#RFclassifier = RandomForestClassifier(n_estimators=200, random_state=0)
#RFclassifier.fit(X_train, y_train)
#predictions = RFclassifier.predict(X_test)
#y_score = RFclassifier.predict_proba(X_test)
#y_score = y_score[:,1]
####

####NN
y_test_new = []
for i in y_test:
    pos = 1
    neg = 0
    if i == 'pos':
        y_test_new.append(pos)
    else:
        y_test_new.append(neg)
y_test = y_test_new
y_train_new = []
for i in y_train:
    pos = 1
    neg = 0
    if i == 'pos':
        y_train_new.append(pos)
    else:
        y_train_new.append(neg)
y_train = y_train_new
max_features = 1500
maxlen = 1500
batch_size = 32
X_train = sequence.pad_sequences(X_train, maxlen=maxlen)
X_test = sequence.pad_sequences(X_test, maxlen=maxlen)
y_train = np.array(y_train)
y_test = np.array(y_test)
model = Sequential()
model.add(Embedding(max_features, 128, input_length=maxlen))
model.add(Bidirectional(LSTM(64)))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))
model.compile('adam', 'binary_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, batch_size=batch_size, epochs=50, validation_data=None)
score = model.evaluate(X_test, y_test)
predictions = model.predict(X_test)
y_score = model.predict_proba(X_test)
####

print(confusion_matrix(y_test,predictions))
print(classification_report(y_test,predictions))
print(accuracy_score(y_test, predictions))

y_true = []

for i in y_test:
    pos = 1
    neg = 0
    if i == 'pos' or i == 1:
        y_true.append(pos)
    else:
        y_true.append(neg)

fpr,tpr,thresold = roc_curve(y_true, y_score)
auc_value = auc(fpr,tpr)
print(auc_value)

###ROC Graph
plt.figure()
lw = 2
plt.figure(figsize=(10,10))
plt.plot(fpr, tpr, color='darkorange',
         lw=lw, label='ROC curve (area = %0.2f)' % auc_value)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic curve')
plt.legend(loc="lower right")
plt.show()
####













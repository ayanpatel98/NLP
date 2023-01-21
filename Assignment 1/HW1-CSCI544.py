#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import nltk
nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('punkt')
import re
from bs4 import BeautifulSoup
import contractions
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Perceptron
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix as cm
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report

 


# In[2]:


# ! pip install bs4 # in case you don't have it installed

# Dataset: https://s3.amazonaws.com/amazon-reviews-pds/tsv/amazon_reviews_us_Beauty_v1_00.tsv.gz


# ## Read Data

# In[3]:


df = pd.read_csv('./amazon_reviews_us_Beauty_v1_00.tsv', sep='\t', error_bad_lines=False, warn_bad_lines=False)


# ## Keep Reviews and Ratings

# In[4]:


df = df[['star_rating', 'review_body']]


# # let ratings with the values of 1 and 2 form class 1, ratings with the value of 3 form class 2, and ratings with the values of 4 and 5 form class 3

# In[5]:


class_one = df[(df['star_rating']==1) | (df['star_rating']==2)]
class_two = df[df['star_rating']==3]
class_three = df[(df['star_rating']==4) | (df['star_rating']==5)]

class_one.loc[:, "label"] =1
class_two.loc[:, "label"] =2
class_three.loc[:, "label"] =3


#  ## We form three classes and select 20000 reviews randomly from each class.
# 
# 

# In[6]:


class_one = class_one.sample(n=20000, random_state=100)
class_two = class_two.sample(n=20000, random_state=100)
class_three = class_three.sample(n=20000, random_state=100)
dataset = pd.concat([class_one, class_two, class_three])
dataset.reset_index(drop=True)
train = dataset.sample(frac=0.8, random_state=100)
test = dataset.drop(train.index)
train = train.reset_index(drop = True)
test = test.reset_index(drop = True)
print(len(train), len(test))


# # Data Cleaning
# 
# 

# # convert all reviews into lowercase

# In[7]:


avg_len_before = (train['review_body'].str.len().sum() + test['review_body'].str.len().sum())/60000

#Covert all reviews to lower case
train['review_body'] = train['review_body'].str.lower()
test['review_body'] = test['review_body'].str.lower()


# # remove the HTML and URLs from the reviews

# In[8]:


# URL Remover code
train['review_body'] = train['review_body'].apply(lambda x: re.split('https:\/\/.*', str(x))[0])
test['review_body'] = test['review_body'].apply(lambda x: re.split('https:\/\/.*', str(x))[0])

def html_tag_remover(review):
    soup = BeautifulSoup(review, 'html.parser')
    review = soup.get_text()
    return review  

train['review_body'] = train['review_body'].apply(lambda review: html_tag_remover(review))
test['review_body'] = test['review_body'].apply(lambda review: html_tag_remover(review))


# # remove non-alphabetical characters

# In[9]:


train['review_body'] = train['review_body'].apply(lambda review: re.sub('[^a-zA-Z]+',' ', review))
test['review_body'] = test['review_body'].apply(lambda review: re.sub('[^a-zA-Z]+',' ', review))


# # remove extra spaces

# In[10]:


train['review_body'] = train['review_body'].apply(lambda review: re.sub(' +', ' ', review))
test['review_body'] = test['review_body'].apply(lambda review: re.sub(' +', ' ', review))


# # perform contractions on the reviews

# In[11]:


def expand_contractions(review):
    review = contractions.fix(review)
    return review

train['review_body'] = train['review_body'].apply(lambda review: expand_contractions(review))
test['review_body'] = test['review_body'].apply(lambda review: expand_contractions(review))
avg_len_after = (train['review_body'].str.len().sum() + test['review_body'].str.len().sum())/60000

print('Average length of the reviews in terms of character length before cleaning: ', avg_len_before)
print('Average length of the reviews in terms of character length after cleaning: ', avg_len_after)


# # Pre-processing

# ## remove the stop words 

# In[12]:


avg_len_before_prepro = (train['review_body'].str.len().sum() + test['review_body'].str.len().sum())/60000
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

def remove_stopwords(review):
    stop_words_english = set(stopwords.words('english'))
    review_word_tokens = word_tokenize(review)
    filtered_review = [word for word in review_word_tokens if not word in stop_words_english]
    return filtered_review

train['review_body'] = train['review_body'].apply(lambda review: remove_stopwords(review))
test['review_body'] = test['review_body'].apply(lambda review: remove_stopwords(review))


# ## perform lemmatization  

# In[13]:


from nltk.stem import WordNetLemmatizer

def review_lemmatize(review):
    lemmatizer = WordNetLemmatizer()
    lemmatized_review = [lemmatizer.lemmatize(word) for word in review]
    return ' '.join(lemmatized_review)    

train['review_body'] = train['review_body'].apply(lambda review: review_lemmatize(review))
test['review_body'] = test['review_body'].apply(lambda review: review_lemmatize(review))

avg_len_after_prepro = (train['review_body'].str.len().sum() + test['review_body'].str.len().sum())/60000

print('Average length of the reviews in terms of character length before preprocessing: ', avg_len_before_prepro)
print('Average length of the reviews in terms of character length after preprocessing: ', avg_len_after_prepro)


# # TF-IDF Feature Extraction

# In[16]:


train_corpus = list(train['review_body'])
test_corpus = list(test['review_body'])
tfidf_vectorizer = TfidfVectorizer(min_df = 0.001)
X_train = tfidf_vectorizer.fit_transform(train_corpus)
X_train = pd.DataFrame(X_train.toarray())
X_test = tfidf_vectorizer.transform(test_corpus)
X_test = pd.DataFrame(X_test.toarray())
Y_train = train['label']
Y_test = test['label']
Y_train = Y_train.astype('int')
Y_test = Y_test.astype('int')


# # Perceptron

# In[21]:


perceptr = Perceptron(random_state = 100)
perceptr.fit(X_train, Y_train)
Y_pred_test = perceptr.predict(X_test)

# Precision of the testing split
precision_test = precision_score(Y_test, Y_pred_test, average=None)
# Recall of the testing split
recall_test = recall_score(Y_test, Y_pred_test, average=None)
# F1-Score of the testing split
f1_test = f1_score(Y_test, Y_pred_test, average=None)
target_names = ['class 1', 'class 2', 'class 3']
print(classification_report(Y_test, Y_pred_test, target_names=target_names))

# Scores
# print('\t    ','CLASS 1','\t', 'CLASS 2','\t', 'CLASS 3')
# print('PRECISION', precision_test[0], precision_test[1], precision_test[2])
# print('RECALL   ', recall_test[0], recall_test[1], recall_test[2])
# print('F1-SCORE ', f1_test[0], f1_test[1], f1_test[2])
# print('Average Scores:')
# print('PRECISION', 'RECALL', 'F1-SCORE')
# print(sum(precision_test)/3, sum(recall_test)/3, sum(f1_test)/3)


# # SVM

# In[18]:


clf_svm = LinearSVC(random_state=100, max_iter=1000)
clf_svm.fit(X_train, Y_train)
Y_pred_test_svm = clf_svm.predict(X_test)

# Precision of the testing split
precision_test = precision_score(Y_test, Y_pred_test_svm, average=None)
# Recall of the testing split
recall_test = recall_score(Y_test, Y_pred_test_svm, average=None)
# F1-Score of the testing split
f1_test = f1_score(Y_test, Y_pred_test_svm, average=None)

# Scores

target_names = ['class 1', 'class 2', 'class 3']
print(classification_report(Y_test, Y_pred_test_svm, target_names=target_names))
# print(precision_test)
# print(recall_test)
# print(f1_test)
# print(sum(precision_test)/3, sum(recall_test)/3, sum(f1_test)/3)


# # Logistic Regression

# In[19]:


logistic = LogisticRegression(random_state = 100, max_iter=1000)
logistic.fit(X_train, Y_train)
Y_pred_test_logis = logistic.predict(X_test)

# Precision of the testing split
precision_test = precision_score(Y_test, Y_pred_test_logis, average=None)
# Recall of the testing split
recall_test = recall_score(Y_test, Y_pred_test_logis, average=None)
# F1-Score of the testing split
f1_test = f1_score(Y_test, Y_pred_test_logis, average=None)

# Scores
target_names = ['class 1', 'class 2', 'class 3']
print(classification_report(Y_test, Y_pred_test_logis, target_names=target_names))
# print(precision_test)
# print(recall_test)
# print(f1_test)
# print(sum(precision_test)/3, sum(recall_test)/3, sum(f1_test)/3)


# # Naive Bayes

# In[20]:


naive_bay = MultinomialNB(force_alpha=True)
naive_bay.fit(X_train, Y_train)
Y_pred_test_naive = logistic.predict(X_test)

# Precision of the testing split
precision_test = precision_score(Y_test, Y_pred_test_naive, average=None)
# Recall of the testing split
recall_test = recall_score(Y_test, Y_pred_test_naive, average=None)
# F1-Score of the testing split
f1_test = f1_score(Y_test, Y_pred_test_naive, average=None)

# Scores
target_names = ['class 1', 'class 2', 'class 3']
print(classification_report(Y_test, Y_pred_test_naive, target_names=target_names))
# print(precision_test)
# print(recall_test)
# print(f1_test)
# print(sum(precision_test)/3, sum(recall_test)/3, sum(f1_test)/3)


import os

import numpy as np
import pandas as pd

import nltk
import re
import string
from nltk import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
from sklearn.naive_bayes import BernoulliNB
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import CountVectorizer

# nltk.download('stopwords')
nltk.download('wordnet')

data_folders_train = ['enron1', 'enron4', 'hw1_train']
data_folders_test = ['enron1', 'enron4', 'hw1_test']

vocabulary = set()

sentences = []
target_vector = []

testing_sentences = []
testing_actual_target = []


def clean_data(text_content):
    tokens = text_content.split()
    regex = re.compile('[%s]' % re.escape(string.punctuation))
    filtered_words = [regex.sub('', w) for w in tokens]
    alpha_numeric_words = [word for word in filtered_words if (word.isalpha())]
    stop_words = set(stopwords.words('english'))
    stop_filtered_words = [w.lower() for w in alpha_numeric_words if w not in stop_words]
    lm = WordNetLemmatizer()
    lemma_words = [lm.lemmatize(word) for word in stop_filtered_words]
    ps = PorterStemmer()
    stemmed_words = [ps.stem(word) for word in lemma_words]
    final = [word for word in stemmed_words if len(word) > 1 and word != 'subject']
    for word in final:
        if word not in vocabulary:
            vocabulary.add(word)
    sentences.append(' '.join(final))
    return final


def update_target_vector(file_obj, target):
    try:
        file_content = file_obj.read()
        target_vector.append(target)
        tokens = clean_data(file_content)
    except UnicodeDecodeError as uni_err:
        print(uni_err)


for folder_name in data_folders_train:
    path = f'./train/{folder_name}/train/ham'
    directory = os.listdir(path)
    for name in directory:
        with open(f"{path}/{name}", "r", encoding="utf-8", errors="ignore") as file:
            update_target_vector(file, 1)

# Parse and tokenize the spam files of all the three datasets

for folder_name in data_folders_train:
    path = f'./train/{folder_name}/train/spam'
    directory = os.listdir(path)
    for name in directory:
        with open(f"{path}/{name}", "r", encoding="utf-8", errors="replace") as file:
            update_target_vector(file, 0)

# save the test samples and actual target output

for folder_name in data_folders_test:
    path = f'./test/{folder_name}/test/ham'
    directory = os.listdir(path)
    for name in directory:
        with open(f"{path}/{name}", "r", encoding="utf-8", errors="replace") as file:
            testing_sentences.append(file.read())
            testing_actual_target.append(1)

for folder_name in data_folders_test:
    path = f'./test/{folder_name}/test/spam'
    directory = os.listdir(path)
    for name in directory:
        with open(f"{path}/{name}", "r", encoding="utf-8", errors="replace") as file:
            testing_sentences.append(file.read())
            testing_actual_target.append(0)

# create word frequency model

cv = CountVectorizer(stop_words='english')
vectors = cv.fit_transform(sentences)
bow_df = pd.DataFrame(vectors.toarray(), columns=cv.get_feature_names())

bow_df.insert(0, 'HAM', target_vector)
bow_df.to_csv('./processed/bagOfWords.csv', index=False)

list_of_words = sorted(cv.vocabulary_.keys())
total_emails = len(sentences)
total_spam = (bow_df['HAM'] == 0).sum()
total_ham = (bow_df['HAM'] == 1).sum()
total_words = len(cv.vocabulary_)

print(total_emails, total_spam, total_ham, total_words)

# Using the same vocabulary to generate Bernoulli model

bernoulli_list = []

for x in sentences:
    temp_y = []
    for w in list_of_words:
        words_set = set(x.split())
        if w in words_set:
            temp_y.append(1)
        else:
            temp_y.append(0)
    bernoulli_list.append(temp_y)

bern_df = pd.DataFrame(data=bernoulli_list, columns=list_of_words)
bern_df.insert(0, 'HAM', target_vector)

bern_df.to_csv('./processed/bernoulli.csv', index=False)


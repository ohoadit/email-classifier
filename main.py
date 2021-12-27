import os
import re
import math
import nltk
import string
import numpy as np
import pickle
import pandas as pd
from nltk import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import CountVectorizer

from multinomialNB import MultinomialNB

nltk.download('omw-1.4')
nltk.download('wordnet')

data_folders_train = ['enron1', 'enron4', 'hw1_train']
data_folders_test = ['enron1', 'enron4', 'hw1_test']

vocabulary = set()

sentences = []
training_target = []

testing_sentences = []
testing_actual_target = []


def clean_data(text_content, save_in_vocab, sentenceList = None):
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
    if save_in_vocab == True:
        for word in final:
            if word not in vocabulary:
                vocabulary.add(word)
    cleaned_sentence = ' '.join(final)
    if sentenceList != None:
        sentenceList.append(cleaned_sentence)
    return cleaned_sentence

def update_training_target(file_obj, target):
    try:
        file_content = file_obj.read()
        training_target.append(target)
        tokens = clean_data(file_content, True, sentences)
    except UnicodeDecodeError as uni_err:
        print(uni_err)

for folder_name in data_folders_train:
    path = f'./train/{folder_name}/train/ham'
    directory = os.listdir(path)
    for name in directory:
        with open(f"{path}/{name}", "r", encoding="utf-8", errors="ignore") as file:
            update_training_target(file, 1)

# Parse and tokenize the spam files of all the three datasets

for folder_name in data_folders_train:
    path = f'./train/{folder_name}/train/spam'
    directory = os.listdir(path)
    for name in directory:
        with open(f"{path}/{name}", "r", encoding="utf-8", errors="replace") as file:
            update_training_target(file, 0)



cv = CountVectorizer(stop_words='english')
vectors = cv.fit_transform(sentences)
x_train = vectors.toarray()
y_train = np.array(training_target)
mnb = MultinomialNB(x_train, y_train, cv.get_feature_names())
mnb.fit()
pickle.dump(mnb, open("model.pkl", "wb"))

for folder_name in data_folders_test:
    path = f'./test/{folder_name}/test/ham'
    directory = os.listdir(path)
    for name in directory:
        with open(f"{path}/{name}", "r", encoding="utf-8", errors="replace") as file:
            clean_data(file.read(), False, testing_sentences)
            testing_actual_target.append(1)

for folder_name in data_folders_test:
    path = f'./test/{folder_name}/test/spam'
    directory = os.listdir(path)
    for name in directory:
        with open(f"{path}/{name}", "r", encoding="utf-8", errors="replace") as file:
            clean_data(file.read(), False, testing_sentences)
            testing_actual_target.append(0)

y_hat = []

for s in testing_sentences:
    y_hat.append(mnb.predict(s))

print(accuracy_score(testing_actual_target, y_hat))
import os

import pandas as pd
import math
import re
import string
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
from sklearn.naive_bayes import BernoulliNB
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

data_folders_test = ['enron1', 'enron4', 'hw1_test']
testing_sentences = []
testing_actual_target = []


def clean_testing_data(text_content):
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
    testing_sentences.append(' '.join(final))
    return final

for folder_name in data_folders_test:
    path = f'./test/{folder_name}/test/ham'
    directory = os.listdir(path)
    for name in directory:
        with open(f"{path}/{name}", "r", encoding="utf-8", errors="replace") as file:
            clean_testing_data(file.read())
            testing_actual_target.append(1)

for folder_name in data_folders_test:
    path = f'./test/{folder_name}/test/spam'
    directory = os.listdir(path)
    for name in directory:
        with open(f"{path}/{name}", "r", encoding="utf-8", errors="replace") as file:
            clean_testing_data(file.read())
            testing_actual_target.append(0)

df = pd.read_csv('./processed/bernoulli.csv')
y_target = df['HAM'].to_list()

total_emails = len(y_target)
total_spam = (df['HAM'] == 0).sum()
total_ham = (df['HAM'] == 1).sum()

list_of_words = sorted(df.columns[~df.columns.isin(['HAM'])].values)
total_words = len(list_of_words)

p_w_spam = {}
p_w_ham = {}
p_spam = total_spam / total_emails
p_ham = total_ham / total_emails
alpha = 1  # laplace smoothing param to avoid zero probability
df_ham = df.loc[df['HAM'] == 1].drop(columns=['HAM'])
df_spam = df.loc[df['HAM'] == 0].drop(columns=['HAM'])
df_ham.loc['occurrence'] = df_ham.sum(axis=0)
df_spam.loc['occurrence'] = df_spam.sum(axis=0)
# total_words_spam = df_spam.loc['occurrence'].sum()
# total_words_ham = df_ham.loc['occurrence'].sum()

# print(total_words_spam, total_words_ham)


def classify_email(p_spam, p_ham):  # argmax function
    if p_spam > p_ham:
        return 0
    return 1


def train_bernoulli_nb():
    df_oc_ham = df_ham.iloc[total_ham]  # access the last row using the index
    df_oc_spam = df_spam.iloc[total_spam]
    for word in list_of_words:
        p_w_ham[word] = (df_oc_ham[word] + alpha) / (total_ham + 2)
        p_w_spam[word] = (df_oc_spam[word] + alpha) / (total_spam + 2)


def test_bernoulli_nb():
    y_hat = []
    for sentence in testing_sentences:
        words = set(sentence.split())
        prob_spam = math.log(p_spam)
        prob_ham = math.log(p_ham)
        for w in list_of_words:
            if w in words:  # if word is present in the sentence and we have the probability
                prob_spam += math.log(p_w_spam[w])
                prob_ham += math.log(p_w_ham[w])
            else:  # if word not present in the sentence apply 1 - p(word)
                prob_spam += math.log(1 - p_w_spam[w])
                prob_ham += math.log(1 - p_w_ham[w])
        y_hat.append(classify_email(prob_spam, prob_ham))
    return y_hat


train_bernoulli_nb()
y_predicted = test_bernoulli_nb()


print(f'Precision: {precision_score(testing_actual_target, y_predicted)*100}')
print(f'Recall: {recall_score(testing_actual_target, y_predicted)*100}')
print(f'F-1: {f1_score(testing_actual_target, y_predicted) * 100}')
print(f'Accuracy: {accuracy_score(testing_actual_target, y_predicted)*100}')

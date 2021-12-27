import math
import pandas as pd


class MultinomialNB():
    def __init__(self, x_train, y_train, columns) -> None:     
        self.alpha = 1
        self.p_w_ham = {}
        self.p_w_spam = {}
        self.x_train = x_train
        self.y_train = y_train

        self.df = pd.DataFrame(x_train, columns=columns)
        self.df.insert(0, 'HAM', y_train)

        total_spam = (self.df['HAM'] == 0).sum()
        total_ham = (self.df['HAM'] == 1).sum()
        total_emails = y_train.shape[0]
        self.total_words = x_train.shape[1]
        self.words = sorted(columns)
        self.p_spam = total_spam / total_emails
        self.p_ham = total_ham / total_emails

        self.df_ham = self.df.loc[self.df['HAM'] == 1].drop(columns=['HAM']).sum(axis=0)
        self.df_spam = self.df.loc[self.df['HAM'] == 0].drop(columns=['HAM']).sum(axis=0)
        self.total_spam_occurrences = self.df_spam.sum()
        self.total_ham_occurrences = self.df_ham.sum()
        self.df = None

    def classify_email(self, prob_spam, prob_ham):  # argmax function
        if prob_spam > prob_ham:
            return 0
        return 1

    def fit(self):
        for word in self.words:
            self.p_w_ham[word] = (self.df_ham[word] + self.alpha) / (self.total_ham_occurrences + (self.total_words * self.alpha))
            self.p_w_spam[word] = (self.df_spam[word] + self.alpha) / (self.total_spam_occurrences + (self.total_words * self.alpha))

    def predict(self, text_content):
        words = text_content.split()
        prob_spam = math.log(self.p_spam)
        prob_ham = math.log(self.p_ham)
        for w in words:
            if w in self.p_w_ham:  # if word is present in the vocabulary and we have the probability
                prob_spam += math.log(self.p_w_spam[w])
                prob_ham += math.log(self.p_w_ham[w])
            else:  # if word not present apply laplace transform to nuke zero probability
                prob_spam += math.log((0 + self.alpha) / (self.total_spam_occurrences + (self.total_words * self.alpha)))
                prob_ham += math.log((0 + self.alpha) / (self.total_ham_occurrences + (self.total_words * self.alpha)))
        return self.classify_email(prob_spam, prob_ham)


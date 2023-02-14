from sklearn.model_selection import train_test_split
from collections import defaultdict
from tqdm import tqdm
import math
import re

def tokenize(text):
    return set(re.findall("[a-z0-9']+" ,text.lower()))

class NaiveBayes():
    def __init__(self, k = 0.5):
        self.k = k # Fator de suavização

        self.tokens = set()
        self.token_spam_counts = defaultdict(int)
        self.token_ham_counts = defaultdict(int)
        self.spam_messages = self.ham_messages = 0


    def train(self, messages):
        for message in messages:
            if message.is_spam:
                self.spam_messages += 1
            else:
                self.ham_messages += 1

            for token in tokenize(message.text):
                self.tokens.add(token)

                if message.is_spam:
                    self.token_spam_counts[token] += 1
                else:
                    self.token_ham_counts[token] += 1

    def probabilities(self, token):
        # Retorna P(Token | Spam) e P(Token | Ham)
        spam = self.token_spam_counts[token]
        ham = self.token_ham_counts[token]
        p_token_spam = (spam + self.k) / (self.spam_messages + 2 * self.k)
        p_token_ham = (ham + self.k) / (self.ham_messages + 2 * self.k)

        return p_token_spam, p_token_ham

    def predict(self, text):
        text_tokens = tokenize(text)
        log_prob_if_spam = log_prob_if_ham = 0

        for token in self.tokens:
            prob_if_spam, prob_if_ham = self.probabilities(token)
            
            if token in text_tokens:
                log_prob_if_ham += math.log(prob_if_ham)
                log_prob_if_spam += math.log(prob_if_spam)
            else:
                log_prob_if_ham += math.log(1.0 - prob_if_ham)
                log_prob_if_spam += math.log(1.0 - prob_if_spam)
        
        prob_if_spam = math.exp(log_prob_if_spam)
        prob_if_ham = math.exp(log_prob_if_ham)

        return prob_if_spam / (prob_if_spam + prob_if_ham)

from collections import namedtuple

class Message():
    def __init__(self, text, is_spam):
        self.text = text
        self.is_spam = is_spam
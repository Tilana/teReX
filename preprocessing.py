from sklearn.feature_extraction import stop_words
import pandas as pd
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer

stopwords = set(stop_words.ENGLISH_STOP_WORDS)
stopwords.update([',', '"', "'", '?', '!', ':', ';', '(', ')', '[', ']', '{', '}'])     

WordNet = WordNetLemmatizer()

def split(text):
    return [word_tokenize(sentence.strip()) for sentence in text]

def lemmatize(tokens):
    return [WordNet.lemmatize(WordNet.lemmatize(word), 'v') for word in tokens]

def lowerAndTokenize(text):
    return word_tokenize(text.lower())

def removeStopwords(tokens):
    return [word for word in tokens if word not in stopwords]

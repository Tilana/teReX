from sklearn.feature_extraction import stop_words
import pandas as pd
import re
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer

stopwords = set(stop_words.ENGLISH_STOP_WORDS)
stopwords.update([',', '"', "'", '?', '!', ':', ';', '(', ')', '[', ']', '{', '}'])     

WordNet = WordNetLemmatizer()
regex = re.compile('[a-zA-Z]+')

def split(text):
    return [word_tokenize(sentence.strip()) for sentence in text]

def lemmatize(tokens):
    return [WordNet.lemmatize(WordNet.lemmatize(word), 'v') for word in tokens]

def tokenizeSentence(sentence):
    return regex.findall(sentence)

def lemmatizeAll(word):
    return WordNet.lemmatize(WordNet.lemmatize(WordNet.lemmatize(word), 'v'), 'a')

def lowerAndTokenize(text):
    return word_tokenize(text.lower())

def removeStopwords(tokens):
    return [word for word in tokens if word not in stopwords]

def standardPreprocessing(data, filename):
    data['tokens'] = data['text'].apply(lowerAndTokenize) 
    
    data['cleanText'] = data['tokens'].apply(lemmatize) 
    data['cleanText'] = data['cleanText'].apply(removeStopwords) 
    data['cleanText'] = data['cleanText'].apply(' '.join)
                                                                  
    data['sentences'] = data['cleanText'].apply(sent_tokenize)
    data['sentences'] = data['sentences'].apply(split)
                                                      
    data.to_pickle(filename)

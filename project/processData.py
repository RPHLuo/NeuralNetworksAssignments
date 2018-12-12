import numpy as np
import pandas as pd
import re

def getVectorDictionary(dim):
    word_vector = pd.read_csv('glove/glove.twitter.27B.{}d.txt'.format(dim),sep=' ', encoding='latin-1')
    word_vector = np.array(word_vector)
    vector_dictionary = {}
    for i in range(0,len(word_vector)):
        vector_dictionary[word_vector[i][0]] = word_vector[i][1:dim+1]
    return vector_dictionary
    #http://nlp.stanford.edu/data/glove.twitter.27B.zip

def getData():
    tr_data = pd.read_csv('all/train.csv',sep=',', encoding='latin-1')
    te_data = pd.read_csv('all/test.csv',sep=',', encoding='latin-1')
    tr_data = np.array(tr_data)
    te_data = np.array(te_data)
    trX = tr_data[:1]
    trY = tr_data[:2]
    teX = te_data[:1]
    teY = te_data[:2]
    trX = np.array([np.array([processWord(word) for word in tweet.split(' ')]] for tweet in trX)
    return trX, trY, teX, teY

def processWord(word):
    word = word.lower()
    REGEX_USER = re.compile('\@\w+')
    REGEX_LINK = re.compile('https?:\/\/[^\s]+')
    REGEX_HTML_ENTITY = re.compile('\&\w+')
    REGEX_NON_ACSII = re.compile('[^\x00-\x7f]')
    REGEX_NUMBER = re.compile(r'[-+]?[0-9]+')


    # Replace ST "entitites" with a unique token
    text = re.sub(REGEX_USER, '<user>', word)
    text = re.sub(REGEX_NUMBER, '<number>', word)
    text = re.sub(REGEX_LINK, '<link>', word)
    <hashtag>
    <url>
    <allcaps>
    <elong>
    <smile>
    <neutralface>
    text = re.sub(REGEX_USER, '<repeat>', word)
    # Remove extraneous text data
    text = re.sub(REGEX_HTML_ENTITY, "", word)
    text = re.sub(REGEX_NON_ACSII, "", word)
    text = re.sub(REGEX_PUNCTUATION, "", word)
    # Tokenize and remove < and > that are not in special tokens
    words = " ".join(token.replace("<", "").replace(">", "")
                     if token not in ['<TICKER>', '<USER>', '<LINK>', '<PRICE>', '<NUMBER>']
                     else token
                     for token
                     in text.split())
    return

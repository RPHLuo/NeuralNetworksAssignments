import numpy as np
import pandas as pd
import re

def getVectorDictionary(dim):
    word_vector = pd.read_csv('glove/glove.twitter.27B.{}d.txt'.format(dim),sep=' ', encoding='latin-1')
    word_vector = np.array(word_vector)
    vector_dictionary = {}
    vocab = []
    emb = []
    tags = []
    isascii = lambda s: len(s) == len(s.encode())
    for i in range(0,len(word_vector)):
        if word_vector[i][0][0] == '<':
            tags.append(word_vector[i][0])
        if isascii(word_vector[i][0]):
            vocab.append(word_vector[i][0])
            emb.append(word_vector[i][1:])
            vector_dictionary[word_vector[i][0]] = word_vector[i][1:dim+1]
    return vector_dictionary, vocab, emb
    #http://nlp.stanford.edu/data/glove.twitter.27B.zip

def getData():
    tr_data = pd.read_csv('all/train.csv',sep=',', encoding='latin-1')
    #te_data = pd.read_csv('all/test.csv',sep=',', encoding='latin-1')
    tr_data = np.array(tr_data)
    trX = tr_data[:,2]
    trY = tr_data[:,1]
    trX = np.array([processTweet(tweet) for tweet in trX])
    return trX, trY, #teX, #teY

def processTweet(tweet):
    REGEX_USER = re.compile('\@\w+')
    REGEX_HASHTAG = re.compile('#')
    REGEX_URL = re.compile('https?:\/\/[^\s]+')
    REGEX_HTML_ENTITY = re.compile('\&\w+')
    REGEX_NON_ACSII = re.compile('[^\x00-\x7f]')
    REGEX_NUMBER = re.compile('[-+]?[0-9]+')
    REGEX_TIME = re.compile('%H:%M')
    REGEX_RETWEET = re.compile('(?<!RT\s)@\S+')
    REGEX_NEUTRALFACE = re.compile(':-\||=p|:-p')
    REGEX_SMILE = re.compile('@-\)|\(:|:\)|:-D|:D|xD|x\)|;\]')
    REGEX_SIGH = re.compile(':-/')
    REGEX_HEART = re.compile('xox')
    REGEX_SADFACE = re.compile("T_T|:'\(|=\(|:\(\)|:\[")

    # Replace ST "entitites" with a unique token
    tweet = re.sub(REGEX_USER, '<user>', tweet)
    tweet = re.sub(REGEX_NUMBER, '<number>', tweet)
    tweet = re.sub(REGEX_URL, '<url>', tweet)
    tweet = re.sub(REGEX_NON_ACSII, "", tweet)
    tweet = re.sub(REGEX_HASHTAG, "<hashtag> ", tweet)
    tweet = re.sub(REGEX_TIME, "<time>", tweet)
    tweet = re.sub(REGEX_RETWEET, "<retweet>", tweet)
    tweet = re.sub(REGEX_SADFACE, "<sadface>", tweet)
    tweet = re.sub(REGEX_NEUTRALFACE, "<neutralface>", tweet)
    tweet = re.sub(REGEX_SMILE, "<smile>", tweet)
    tweet = re.sub(REGEX_SIGH, "<sigh>", tweet)
    tweet = re.sub(REGEX_HEART, "<heart>", tweet)
    tweet = re.sub(r"\'s", " is", tweet)
    tweet = re.sub(r"\'ve", " have", tweet)
    tweet = re.sub(r"n\'t", " not", tweet)
    tweet = re.sub(r"\'re", " are", tweet)
    tweet = re.sub(r"\'d", " had", tweet)
    tweet = re.sub(r"\'ll", " will", tweet)

    # Remove extraneous text data
    tweet = re.sub(REGEX_HTML_ENTITY, "<unknown>", tweet)
    tweet = re.sub(REGEX_NON_ACSII, "<unknown>", tweet)
    tweet = tweet.strip()
    return tweet

import pandas as pd 
import numpy as np
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from collections import Counter
import string
import re

# download required library from nltk
# nltk.download('stopwords')
# nltk.download('punkt')

# create stemmer
factory = StemmerFactory()
stemmer = factory.create_stemmer()

stop_words_ind = list(stopwords.words('indonesian'))
stop_words_eng = list(stopwords.words('english'))
stop_words_custom = ['kau', 'yg', 'mcm', 'gak', 'nak', 'ni', 'tu', 'la', 'je', 'kat', 'ya', 'dgn', 'tau', 'org', 'rt', 'aja', 'nk', 'dah',
                        'orang', 'sy', 'ga', 'kalo', 'kena']
stop_words = np.unique(stop_words_ind+stop_words_eng+stop_words_custom)

def text_preprocessing(text):

    # remove numbers
    text = re.sub(r'\d+', '', text)
    # remove links
    text = re.sub('http[s]?://\S+', '', text)

    # tokennization
    tokens = word_tokenize(text)

    # lemmetization and remove punctuation
    words = []
    for token in tokens:
        if token not in string.punctuation:
            temp = stemmer.stem(token)
            words.append(temp)

    # remove stopwords
    cleaned = []
    for word in words:
        if word not in stop_words:
            cleaned.append(word)

    # traverse in the string     
    complete_sentence = ' '.join([str(word) for word in cleaned])
    
    return complete_sentence
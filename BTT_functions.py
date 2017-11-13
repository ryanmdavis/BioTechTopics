import string,nltk
from nltk.corpus import stopwords
from nltk.stem.porter import *

def tokenizeAndStemStrings(text):
    
    # turn text to tokens
    tokens = nltk.word_tokenize(text)
    
    # remove stop words
    tokens_no_sw = [word for word in tokens if not word in stopwords.words('english')]
    
    # stem words
    stemmed = []
    stemmer = PorterStemmer()
    for item in tokens_no_sw:
        # this line converts strings to unicode, so here I do it explicitly
        try:
            stemmed.append(stemmer.stem(unicode(item)))
        except:
            stemmed.append(unicode(item)) # for example, stemmer can't stem aed because it expects a letter before a
            print "stemmer skipped word: " + str(unicode(item))
    
    return stemmed

def cleanString(x):
    return x.encode('ascii','ignore').translate(None, string.punctuation).replace('\n', ' ').replace('\r', ' ')
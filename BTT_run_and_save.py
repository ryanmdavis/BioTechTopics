# note to self.  Have BioTechTopics pickle the key objects and matricies in different files
from sklearn.feature_extraction.text import TfidfVectorizer
#import nltk
from nltk.stem.porter import *
#from nltk.corpus import stopwords
import pickle
from BioTechTopics import Topics
from BTT_functions import tokenizeAndStemStrings,cleanString
#from sklearn.decomposition import LatentDirichletAllocation
import time

if __name__ == '__main__':
    print('executed main')

    t=Topics()
    t.getText(json_file_loc='/home/ryan/Dropbox/Code/DataIncubatorChallenge/BioTechTopics/data/all_reports.json')
    #t.processCorpus()
    
    # train and save tfidf representation: 20 minutes
    print('\nTraining tf-idf Vectorizer')
    start = time.time()
<<<<<<< HEAD
    tfidf_vectorizer = TfidfVectorizer(tokenizer=tokenizeAndStemStrings, stop_words='english',ngram_range=(1,2), use_idf=True, smooth_idf = False, norm=None, min_df=0.002, max_df=0.2)    
=======
    tfidf_vectorizer = TfidfVectorizer(tokenizer=tokenizeAndStemStrings, stop_words='english',ngram_range=(1,1), use_idf=True, smooth_idf = False, norm=None, min_df=0.002, max_df=0.2)    
>>>>>>> 08889f49ce4f5aef81cbfd3442d69fd20d61346e
    tfidf = tfidf_vectorizer.fit_transform(t.text_df['text_body'].apply(cleanString))
    end = time.time()
    print('Done training after ' + str(end-start) + ' seconds')
    
    with open('./data/tfidf_vectorizer_maxdf0_2.p', 'wb') as f:
        pickle.dump(tfidf_vectorizer,f)
    with open('./data/tfidf_maxdf0_2.p', 'wb') as f:
        pickle.dump(tfidf,f)
    
    del tfidf_vectorizer,tfidf

print("done")
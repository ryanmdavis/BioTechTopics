import os, json, string, nltk, sys, inspect
from nltk.stem.porter import *
from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from nltk.corpus import stopwords
from sklearn.metrics.pairwise import linear_kernel
from summa import keywords
import numpy as np
from collections import Counter
import operator
import itertools
import pandas as pd
import time
from wordcloud import WordCloud
import matplotlib.pyplot as plt

class Topics(object):
    
    def __init__(self):
        print 'Reading corpus'
        # getText opens up a pandas dataframe, extracts text, cleans it (removes punctuation, \n,\r, and makes lowercase), stores it in Topics attributes
        self.text_list, self.raw_text_list, self.text_keywords = self.getText('/home/ryan/Dropbox/Code/DataIncubatorChallenge/BioTechTopics/data.json')
        print 'Corpus contains ' + str(len(self.text_list)) + ' unique files'
        
        # train count vectorizer to do LDA
        start = time.time()
        print '\nTraining Count Vectorizer'
        self.tf, self.tf_vectorizer, self.feature_names=self.trainCountVectorizer(self.text_list)
        end = time.time()
        print 'Done training after ' + str(end-start) + ' seconds'
        
        # train tfidf vectorizer to do LDA
        start = time.time()
        print '\nTraining tfidf Vectorizer'
        self.createTfidfTable(self.text_list)
        end = time.time()
        print 'Done training after ' + str(end-start) + ' seconds'        
        
        # LDA
        start = time.time()
        print '\nTraining LDA probability distributions'
        self.performLDA(10)
        end = time.time()
        print 'Done training after ' + str(end-start) + ' seconds'
        
        # calculate wordclout text for topics
        self.calcWordCloudText(25)
        
        # set the wordcloud titles
        self.topic_titles=['Blood tests and healthcare payers','Microbiome and Bacteria','New Drugs','Clinical Drug Development','Immunotherapies',' ','Health Data',' ','Devices and Diagnostics','Topic 9',]
    
        print '\nTopics instance ready'
    # code from https://github.com/amueller/word_cloud/blob/master/examples/simple.py
    def showTopicWordCloud(self,topic_number,fs=(6,4)):
        
        if topic_number<0:
            topic_number=0
        elif topic_number>self.num_topics-1:
            topic_number=self.num_topics-1       
        
        # Generate a word cloud image
        wordcloud = WordCloud(max_words=self.n_top_words).generate(self.topic_words_text[topic_number])
        
        # Display the image:
        plt.figure(figsize=fs)
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis("off")
        plt.title(self.topic_titles[topic_number])
        plt.show()
         
        
    def performLDA(self,n_topics):
        self.lda = LatentDirichletAllocation(n_topics, max_iter=30, learning_method='online', learning_offset=50., random_state=0)
        self.lda.fit(self.tf)
        self.num_topics=n_topics
    
    def printTopWords(self,n_top_words): #from scikit-learn.org
        for topic_idx, topic in enumerate(self.lda.components_):
            message = "Topic #%d: " % topic_idx 
            message+= " " + self.topic_titles[topic_idx].upper() + ": "
            message += ", ".join([self.feature_names[i]
                                 for i in topic.argsort()[:-n_top_words - 1:-1]])
            print("\n" + message)
        print()
    
    def calcWordCloudText(self,n_top_words):
        topic_words_text=[]
        for topic_idx, topic in enumerate(self.lda.components_):
            topic_words = [self.feature_names[i] for i in topic.argsort()[:-n_top_words - 1:-1]]
            posterior_word_prob = [topic[i] for i in topic.argsort()[:-n_top_words - 1:-1]]
            
            # take the matrix product of the term-doc matrix and the vector representation of each topic word.
            # Sum the resulting vector to get the word count
            # Also, multiply by posterior word probability to scale the word size by P(word|topic)
            topic_word_frequency = [sum(np.dot(self.tf,self.tf_vectorizer.transform([topic_word]).transpose()))[0,0] for topic_word in topic_words]
            topic_word_frequency = [int(topic_word_frequency[i]*posterior_word_prob[i]) for i in range(len(topic_word_frequency))]
            
            text=''
            for tw_num in range(len(topic_words)):
                text = text + (topic_words[tw_num] + ' ')*topic_word_frequency[tw_num]
            topic_words_text.append(text)
        self.topic_words_text = topic_words_text
        self.n_top_words=n_top_words
        
    def stemKeywords(self,keyword_list):
        stemmer=PorterStemmer()
        for doc_num in range(len(keyword_list)):
            for keyword_num in range(len(keyword_list[doc_num])):
                keyword_list[doc_num][keyword_num]=' '.join([stemmer.stem(word) for word in nltk.word_tokenize(keyword_list[doc_num][keyword_num])])
                pass
        return keyword_list
    
    def getText(self,json_file_loc):
        # parse all of the JSON objects in the file.
        text_df = pd.read_json(json_file_loc)
        cleanString = lambda x:x.lower().encode('ascii','ignore').translate(None, string.punctuation).replace('\n', ' ').replace('\r', ' ')
        text_body=list(text_df['text_body'].apply(cleanString))
        raw_text_body=list(text_df['text_body'])
        text_keywords=list(text_df['keywords'])
        return text_body, raw_text_body, text_keywords
    
    def tokenizeAndStemStrings(self,text):
        
        # turn text to tokens
        tokens = nltk.word_tokenize(text)
        
        # remove stop words
        tokens_no_sw = [word for word in tokens if not word in stopwords.words('english')]
        
        # stem words
        stemmed = []
        stemmer = PorterStemmer()
        for item in tokens_no_sw:
            # this line converts strings to unicode, so here I do it explicitly
            stemmed.append(stemmer.stem(unicode(item)))
        
        return stemmed
        #return tokens_no_sw
    
    def trainCountVectorizer(self,text_list):
                    
        tf_vectorizer = CountVectorizer(tokenizer=self.tokenizeAndStemStrings, stop_words='english',ngram_range=(1,2),max_df=0.95, min_df=2) #max_features=100        
        tf = tf_vectorizer.fit_transform(text_list)
        feature_names=tf_vectorizer.get_feature_names()
    
        return tf, tf_vectorizer, feature_names
    
    def createTfidfTable(self,text_list):
                    
        self.tfidf_vectorizer = TfidfVectorizer(tokenizer=self.tokenizeAndStemStrings, stop_words='english',ngram_range=(1,4), use_idf=True, smooth_idf = False, norm=None)
        self.tfidf = self.tfidf_vectorizer.fit_transform(text_list)
    
        #return tfs, tfidf

    def getKeywordsByTextRank(self,text_list, num_keywords_to_choose_array, syntactic_filter=None):
    
        keywords_by_TextRank=[[]]*len(text_list)
    
        #for doc_num in range(len(text_list)):
        #    tokenized_and_stemmed=' '.join(tokenizeAndStemStrings(text_list[doc_num]))
        #    kw=keywords.keywords(tokenized_and_stemmed)
        #    keywords_by_TextRank[doc_num]=keywords.keywords(tokenized_and_stemmed).split('\n')[0:num_keywords_to_generate_array[doc_num]]
            
        for doc_num in range(len(text_list)):
            
            if syntactic_filter !=None:
                text_filtered = self.syntacticFilter(text_list[doc_num],syntactic_filter)
            else:
                text_filtered=text_list[doc_num]
            # run the TextRank algorithm
            # kw=keywords.keywords(text_filtered, ratio=1, split=True)
            kw=self.summaKeywords(text_filtered,text_list[doc_num], ratio=0.2, words=None, language="english", split=False, scores=False)
            kw_list=kw.split("\n")
            
            # tokenize the result
            kw_ts=[' '.join(self.tokenizeAndStemStrings(kw_list[x])) for x in range(len(kw_list))]
            
            # determine number of keywords to generate
            if num_keywords_to_choose_array[doc_num] >= 0:
                num_keywords_to_choose=num_keywords_to_choose_array[doc_num]
            else:
                num_keywords_to_choose=int(max(len(kw_ts)/abs(num_keywords_to_choose_array[doc_num]),1))
            
            # remove duplicate keywords caused by stemming, and only return the specified number
            doc_keywords=[]
            for kw_num in range(len(kw_ts)):
                if kw_ts[kw_num] not in kw_ts[kw_num+1:]:
                    doc_keywords.append(kw_ts[kw_num])
            
                if len(doc_keywords) == num_keywords_to_choose:
                    break
                
            keywords_by_TextRank[doc_num]=doc_keywords
        return keywords_by_TextRank
    
    def summaKeywords(self,text_filtered,text_original, ratio=0.2, words=None, language="english", split=False, scores=False):
        # Gets a dict of word -> lemma
        tokens = keywords._clean_text_by_word(text_filtered, language)
        split_text = list(keywords._tokenize_by_word(text_filtered))
    
        # Creates the graph and adds the edges
        graph = keywords._build_graph(keywords._get_words_for_graph(tokens))
        keywords._set_graph_edges(graph, tokens, split_text)
        del split_text # It's no longer used
    
        keywords._remove_unreachable_nodes(graph)
    
        # Ranks the tokens using the PageRank algorithm. Returns dict of lemma -> score
        pagerank_scores = keywords._pagerank(graph)
    
        extracted_lemmas = keywords._extract_tokens(graph.nodes(), pagerank_scores, ratio, words)
    
        lemmas_to_word = keywords._lemmas_to_words(tokens)
        keywords_out = keywords._get_keywords_with_score(extracted_lemmas, lemmas_to_word)
    
        # text.split() to keep numbers and punctuation marks, so separeted concepts are not combined
        combined_keywords = keywords._get_combined_keywords(keywords_out, text_original.split())
    
        return keywords._format_results(keywords_out, combined_keywords, split, scores)

    def syntacticFilter(self,text,reg_exp_filter):
        
        # tokenize and tag
        tokens = nltk.word_tokenize(text)
        tagged = nltk.pos_tag(tokens) # part of speech tagging, required for searching for grammars
        
        # create parser
        cp=nltk.RegexpParser(reg_exp_filter)
        
        # parse to find works assoated with the reg_exp_filter
        tree=cp.parse(tagged)
        
        # select only words that satisfy filter
        filtered_text=[]
        for subtree in tree.subtrees(lambda t: t.label()=='Filt'):
            filtered_text.append(subtree.leaves()[0][0])
        
        return_text=' '.join(filtered_text)
        return return_text
    
    def getRegExps(self,grammar):
        cp=nltk.RegexpParser(grammar)
        regexp_list=[[]]*len(self.text_list)
        num_regexp_list=[0]*len(self.text_list)
        for doc_num in range(len(self.text_list)):
            tokens = nltk.word_tokenize(self.text_list[doc_num])
            tagged = nltk.pos_tag(tokens) # part of speech tagging, required for searching for grammars
            tree=cp.parse(tagged)
            this_list=[]
            for subtree in tree.subtrees():
                if subtree.label() == 'KW': 
                    this_list.append(' '.join([subtree.leaves()[x][0] for x in range(len(subtree.leaves()))]))
            regexp_list[doc_num]=this_list
            num_regexp_list[doc_num]=len(this_list)
            
        return regexp_list,num_regexp_list

    #tree traversal from http://www.nltk.org/book/ch07.html     
    def traverse_ne(self,t,use_next=0,ne_type='PERSON',ne_list=[]):
        try:
            t.label()
        except AttributeError:
            if use_next==1:
                ne_list.append(t[0])
                use_next=0
            #print(t)
        else:
            # Now we know that t.node is defined
            #print('(', t.label())
            if ne_type==t.label():
                use_next=1
            for child in t:
                ne_list=self.traverse_ne(child,use_next,ne_type,ne_list)
            #print(')')
        return ne_list

    def traverse_np(self,t,np_list=[]):
        try:
            t.label()
        except AttributeError:
            return np_list
        else:
            if t.label()=='NP':
                np_list.append(' '.join([t[x][0] for x in range(len(t))]))
            for node in t:
                np_list = self.traverse_np(node,np_list)
        return np_list

    def getKeywordsByTfidf(self,regexps_list,num_keywords_to_choose_array):
        # Inputs:
        #    num_keywords_to_choose_array: if positive then function will return that many keywords, if negative value L then return len(regexps_list)/|L| keywords
    
        
        # allocate some memory for keywords
        keywords = [[]]*len(regexps_list)
        
        # define stemmer
        stemmer=PorterStemmer()
        
        # get feature names (i.e. potential keywords) and assign idf to a dictionary
        feature_names=self.tfidf_vectorizer.get_feature_names()
        idf=self.tfidf_vectorizer.idf_-1
        idf_dict={feature_names[x]:idf[x] for x in range(len(feature_names))}
        
        for doc_num in range(len(regexps_list)):
            # calculate term frequencies
            c=Counter(regexps_list[doc_num])
            term_list=c.most_common()
            # tf_dict={' '.join([stemmer.stem(w) for w in nltk.word_tokenize(term_list[x][0]) if not w in stopwords.words('english')]): term_list[x][1] for x in range(len(term_list))}
            tf_dict={' '.join([stemmer.stem(w) for w in nltk.word_tokenize(term_list[x][0]) if not w in stopwords.words('english')]): term_list[x][1] for x in range(len(term_list)) if term_list[x][0] in idf_dict}
        
            # now calculate tf-idf
            tfidf_dict={key: idf_dict[key]*tf_dict[key] for key in tf_dict.keys()}
            if num_keywords_to_choose_array[doc_num] >= 0:
                num_keywords_to_choose=num_keywords_to_choose_array[doc_num]
            else:
                num_keywords_to_choose=int(max(len(tfidf_dict)/abs(num_keywords_to_choose_array[doc_num]),1))
            sorted_tfidf_keys=sorted(tfidf_dict.iteritems(),key=operator.itemgetter(1),reverse=True)[0:num_keywords_to_choose]
            
            keywords[doc_num]=[sorted_tfidf_keys[x][0] for x in range(len(sorted_tfidf_keys))]
            
        return keywords
    
    def showWhosWhoWordCloud(self,ww_list,query): 
        # Generate a word cloud image
        ww_text=" ".join(ww_list)
        wordcloud = WordCloud(max_words=100).generate(ww_text)
        
        # Display the generated image:
        # the matplotlib way:
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis("off")
        plt.title("Who's who for query \"" + query + "\"")
        plt.show()
        print "Who's who:" + " ,".join(ww_list[0:15])

    ## WhosWho? function.  Uses keyword extraction, named entity recognition, and tfidf information retrieval to 
    ## Identify prominent individuals relevant to a user's query
    def ww(self,query,kw_algorithm='tfidf'):
        
        ## set default ascii encoding
        reload(sys)
        sys.setdefaultencoding('utf8')
        
        ## define grammars
        g1='{(<JJ>* <NN.*>+<IN>)? <JJ>* <NN.*>+}'
        g2='{<NN.*|JJ>}'
        g3='{<NN.*>}'
        grammar_tf=g1
        doc_num=0
        
        ## find the most relevant documents by using a tfidf search of the corpus for the query
        ## the algorithm only consider's documents that have a cosine_similarity greater than one tenth of the maximum cos_sim
        ## this ensures that we don't consider documents of minimal relevance.
        response=self.tfidf_vectorizer.transform([query])
        cosine_similarities = linear_kernel(self.tfidf, response).flatten() #here, tfs has the actual tfidf matrix
        sorted_cs_arg=cosine_similarities.argsort()[::-1]
        sorted_cosine_similarities=cosine_similarities[sorted_cs_arg]
        top_docs_indicies=[sorted_cs_arg[x] for x in range(len(sorted_cosine_similarities)) if sorted_cosine_similarities[x]>sorted_cosine_similarities[0]/10]
        top_docs=[self.text_list[x] for x in top_docs_indicies]
        print 'Found ' + str(len(top_docs)) + ' documents relevant to query "' + query + '"'

        ## extract keywords from the top 10 matches from the tfidf query
        if kw_algorithm=='textrank':
            # find the top 10 keywords of each document by textrank
            keyword_list = self.getKeywordsByTextRank(top_docs, [20]*len(top_docs_indicies), 'Filt: {<NN.*>}')
        else:
            # find the top 10 keywords of each document by tf-idf
            self.regexp_list,self.num_regexp_list = self.getRegExps('KW: '+grammar_tf)
            top_docs_regexp=[self.regexp_list[x] for x in top_docs_indicies]
            keyword_list = self.getKeywordsByTfidf(top_docs_regexp, [10]*len(top_docs_indicies))
        
        ## find the named entities for each document
        # remove characters that can't be encoded to ascii
        whos_who=[]
        for doc_num in range(len(top_docs_indicies)):
            #ascii_compatible = self.raw_text_list[top_docs_indicies[0]].encode('utf-8','ignore').decode('utf-8')
            tokens = nltk.word_tokenize(self.raw_text_list[top_docs_indicies[doc_num]].encode('utf-8','ignore').decode('utf-8'))
            tagged = nltk.pos_tag(tokens)
            ne_tree = nltk.chunk.ne_chunk(tagged)
            ne_list = self.traverse_ne(ne_tree, ne_type="PERSON") + self.traverse_ne(ne_tree, ne_type="ORGANIZATION")
            ne_list = list(set(ne_list)) # remove duplicates
         
            ## find the noun phrases for each document
            np_grammar = r"""
                NP: {<DT|PP\$>?<JJ>*<NN.*>+}   # chunk determiner/possessive, adjectives and noun/s
                """
            pnp_grammar  = r"""
                NP: {<NNP>*}   # series of proper nouns
                """   
                
            cp=nltk.RegexpParser(pnp_grammar)
            chunked_text=cp.parse(tagged)
            np_list=self.traverse_np(chunked_text)
            np_list_lower=[np_list[x].lower() for x in range(len(np_list))]
            #ne_generator = nltk.chunk.ne_chunk_sents(chunked_text)
            
            ## determine which keywords are named entities by taking the intersection (AND) of the keyword and named entity list
            # ne_keyword_list=[ne_list[x] for x in range(len(ne_list)) if ne_list[x].lower() in keyword_list[doc_num]] 
            ne_keyword_list=[ne_list[x] for x in range(len(ne_list)) if True in [ne_list[x].lower() in keyword_list[doc_num][y] for y in range(len(keyword_list[doc_num]))]] 
            
            ## retrieve full noun phrase for all named entity keywords
            ne_keyword_phrase_list=[[np_list[x] for x in range(len(np_list)) if ne_keyword_list[y] in np_list[x]] for y in range(len(ne_keyword_list))]
            ne_keyword_phrase_list=list(itertools.chain(*ne_keyword_phrase_list))
            
            ## remove characters that are not alphanumeric or white space
            ne_keyword_phrase_list=[re.search('[\w\s\'-]+',ne_keyword_phrase_list[x]).group() for x in range(len(ne_keyword_phrase_list)) if bool(re.search('[\w\s]+',ne_keyword_phrase_list[x]))]
            ne_list_no_dup=[]
            
            ## removes elements in ne_keyword_phrase_list that are duplicates or substrings of other elements
            ne_keyword_phrase_list=list(set(ne_keyword_phrase_list))
            for x in range(len(ne_keyword_phrase_list)):
                element_excluded=[ne_keyword_phrase_list[y] for y in range(len(ne_keyword_phrase_list)) if y != x]
                duplicate = 0
                for y in range(len(element_excluded)):
                    if ne_keyword_phrase_list[x] in element_excluded[y]:
                        duplicate = 1
                if not duplicate:
                    ne_list_no_dup.append(ne_keyword_phrase_list[x])
                    
            #whos_who=whos_who+ne_list_no_dup
            whos_who=whos_who+ne_list_no_dup
        #problem ne_chunk_sents requires all tokens to be in a phrase (Which I can't do)
        #ne_chunk doesn't let me choose the chunk grammar
        #solution: use ne_chunk to find the ne, and use REgexpParser to give the grammars to find the full noun phrases
        
        whos_who=[word for word in list(whos_who) if query.lower() not in word.lower()]
        if not bool(whos_who):
            print('Unfortunately there are no key people in the corpus matching the query '+query)
        else:
            self.showWhosWhoWordCloud(whos_who,query)
            #print 'Who''s who for the query "' + query + '":'
            #print(whos_who)
        return whos_who            

if __name__ == '__main__':
    print 'executed main'   

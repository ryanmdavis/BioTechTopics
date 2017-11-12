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
#from wordcloud import WordCloud
import matplotlib.pyplot as plt
import pickle
from BTT_functions import *

import imp

# for some reason, Bokeh can't find wordcloud module
#f, filename, description = imp.find_module('wordcloud')
wordcloud = imp.load_module('wordcloud', None, '/home/ryan/Dropbox/Code/Virtual-Environments/ds_py27_env/lib/python2.7/site-packages/wordcloud', ('', '', 5))
from wordcloud import WordCloud

class Topics(object):
    
    def __init__(self):
        plt.ion()
        # self.getText()
        self.topic_titles=['Blood tests and healthcare payers','Microbiome and Bacteria','New Drugs','Clinical Drug Development','Immunotherapies',' ','Health Data',' ','Devices and Diagnostics','Topic 9',]
    
        print '\nTopics instance ready'
    # code from https://github.com/amueller/word_cloud/blob/master/examples/simple.py
    
    def processCorpus(self,save_file_loc='./data/all_data_processed.json'):
        
        # basic progress bar
        df_len=len(self.text_df)
        num_dashes=100
        docs_per_dash=float(df_len)/num_dashes
        dash_num=0
        print("Progress:\n|" + "-"*num_dashes + "|")
        sys.stdout.write("|")
        
        # allocate some memory for new data frame
        new_df= pd.DataFrame({'TR_keywords':['']*df_len,'TR_keyword_scores':['']*df_len,'year':['']*df_len,'month':['']*df_len,'NE':['']*df_len,'FBT_keywords':['']*df_len,'doc_num':['']*df_len,'author':['']*df_len,'title':['']*df_len})
        for doc_num in range(len(self.text_df)):
            if doc_num>docs_per_dash*dash_num:
                sys.stdout.write("-")
                dash_num+=1
            
            try:
                keyword_list = self.getKeywordsByTextRank({str(doc_num):10}, [doc_num], syntactic_filter='Filt: {<NN.*>}')[0]
                #tokens = nltk.word_tokenize(self.text_df['text_body'][doc_num].encode('utf-8','ignore').decode('utf-8'))
                tokens = nltk.word_tokenize(cleanString(self.text_df['text_body'][doc_num]))
                tagged = nltk.pos_tag(tokens)
                ne_list=self.ne(tagged) #named entities
                np_list=self.nps(tagged) #noun phrases
                scored_kw_phrases=self.scoreNeNp(ne_list,np_list,keyword_list)
            
            # now assign the keywords to the dataframe columns
            
                new_df['TR_keywords'][doc_num]=','.join([scored_kw_phrases[x][0] for x in range(len(scored_kw_phrases))])
                new_df['TR_keyword_scores'][doc_num]=','.join([str(scored_kw_phrases[x][1]) for x in range(len(scored_kw_phrases))])
                new_df['NE'][doc_num]=','.join(ne_list)
                new_df['FBT_keywords'][doc_num]=','.join(self.text_df['keywords'][doc_num]).replace(' ','_')
                new_df['doc_num'][doc_num]=doc_num
                new_df['month'][doc_num]=str(self.text_df['date'][doc_num])[5:7]
                new_df['author'][doc_num]=self.text_df['author'][doc_num]
                new_df['year'][doc_num]=float(str(self.text_df['date'][doc_num])[0:4])+float(str(self.text_df['date'][doc_num])[5:7])/12.+float(str(self.text_df['date'][doc_num])[8:10])/30
                new_df['title'][doc_num]=self.text_df['title'][doc_num]
            except:
                new_df['TR_keywords'][doc_num]=''
                new_df['TR_keyword_scores'][doc_num]=''
                new_df['NE'][doc_num]=''
                new_df['year'][doc_num]=0
                new_df['FBT_keywords'][doc_num]=''
                new_df['doc_num'][doc_num]=doc_num
                new_df['month'][doc_num]=''
                new_df['author'][doc_num]=''
                new_df['title'][doc_num]=''
                print "Error at document number " + str(doc_num)
                print("\nProgress:\n|" + "-"*num_dashes + "|")
                sys.stdout.write("|")
                dash_num+=1
                sys.stdout.write("-"*dash_num)
        sys.stdout.write("|\n")
        new_df.to_json('/home/ryan/Dropbox/Code/DataIncubatorChallenge/BioTechTopics/data/all_reports_processed2.json')
        
    def showTopicWordCloud(self,topic_number,fs=(6,4)):
        
        if topic_number<0:
            topic_number=0
        elif topic_number>self.lda.n_topics-1:
            topic_number=self.lda.n_topics-1       
        
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
        feature_names=self.tf_vectorizer.get_feature_names() #LDA was trained on tf_vectorizer representation
        for topic_idx, topic in enumerate(self.lda.components_):
            message = "Topic #%d: " % topic_idx 
            message+= " " + self.topic_titles[topic_idx].upper() + ": "
            message += ", ".join([feature_names[i] for i in topic.argsort()[:-n_top_words - 1:-1]])
            print("\n" + message)
        print()
    
    def calcWordCloudText(self,n_top_words):
        topic_words_text=[]
        feature_names=self.tf_vectorizer.get_feature_names() #LDA was trained on tf_vectorizer representation
        for topic_idx, topic in enumerate(self.lda.components_):
            topic_words = [feature_names[i] for i in topic.argsort()[:-n_top_words - 1:-1]]
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
    
    def getText(self,json_file_loc='./data/all_reports.json'):
        # parse all of the JSON objects in the file.
        self.text_df = pd.read_json(json_file_loc)
        print 'Corpus contains ' + str(self.text_df.shape[0]) + ' unique files'
    
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
    
    def trainCountVectorizer(self):
                    
        self.tf_vectorizer = CountVectorizer(tokenizer=self.tokenizeAndStemStrings, stop_words='english',ngram_range=(1,2),max_df=0.95, min_df=2) #max_features=100        
        self.tf = self.tf_vectorizer.fit_transform(self.text_df['text_body'].apply(cleanString))
        self.feature_names=self.tf_vectorizer.get_feature_names()
    
    def createTfidfTable(self):
                    
        tfidf_vectorizer = TfidfVectorizer(tokenizer=self.tokenizeAndStemStrings, stop_words='english',ngram_range=(1,4), use_idf=True, smooth_idf = False, norm=None)
        tfidf = tfidf_vectorizer.fit_transform(self.text_df['text_body'].apply(cleanString))
    
#         if dump:
#             f = open('./data/tfidf_vectorizer.p', 'w')
#             pickle.dump(tfidf_vectorizer,f)
#             f.close()
#             f = open('./data/tfidf.p', 'w')
#             pickle.dump(tfidf,f)
#             f.close()
        #return tfs, tfidf
        
        #self.tfidf_vectorizer=tfidf_vectorizer
        #self.tfidf=tfidf
        
        return tfidf_vectorizer,tfidf

    def getKeywordsByTextRank(self, num_keywords_to_choose_dict, doc_indicies, syntactic_filter=None):
    
        keywords_by_TextRank=[]
    
        #for doc_num in range(len(text_list)):
        #    tokenized_and_stemmed=' '.join(tokenizeAndStemStrings(text_list[doc_num]))
        #    kw=keywords.keywords(tokenized_and_stemmed)
        #    keywords_by_TextRank[doc_num]=keywords.keywords(tokenized_and_stemmed).split('\n')[0:num_keywords_to_generate_array[doc_num]]
            
        for doc_index in doc_indicies:
            
            if syntactic_filter !=None:
                text_filtered = self.syntacticFilter(cleanString(self.text_df['text_body'][doc_index]),syntactic_filter)
            else:
                text_filtered = cleanString(self.text_df['text_body'][doc_index])
            # run the TextRank algorithm
            kw=self.summaKeywords(text_filtered,self.text_df['text_body'][doc_index].encode('ascii','ignore'), ratio=0.2, words=None, language="english", split=False, scores=True)
            kw_list=[kw[x][0] for x in range(len(kw))]
                     
            # determine number of keywords to generate
            if num_keywords_to_choose_dict[str(doc_index)] >= 0:
                num_keywords_to_choose=num_keywords_to_choose_dict[str(doc_index)]
            else:
                num_keywords_to_choose=int(max(len(kw_list)/abs(num_keywords_to_choose_dict[str(doc_index)]),1))
            
            # remove duplicate keywords caused by stemming, and only return the specified number
            doc_keywords=[]
            for kw_num in range(len(kw)):
                if kw[kw_num] not in kw_list[kw_num+1:]:
                    doc_keywords.append(kw[kw_num])
            
                if len(doc_keywords) == num_keywords_to_choose:
                    break
                
            keywords_by_TextRank.append(doc_keywords)
        #if scores:
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
    
    def getRegExps(self,grammar,doc_indicies):
        cp=nltk.RegexpParser(grammar)
        regexp_list=[[]]*len(doc_indicies)
        num_regexp_list=[0]*len(doc_indicies)
        for doc_num in range(len(doc_indicies)):
            tokens = nltk.word_tokenize(self.text_df['text_body'][doc_num])
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
            t.label() #i.e. is t a tree?
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
            t.label() #i.e. is t a tree?
        except AttributeError:
            return np_list
        else:
            if t.label()=='NP':
                np_list.append(' '.join([t[x][0] for x in range(len(t))]))
            for node in t:
                np_list = self.traverse_np(node,np_list)
        return np_list

#     def getKeywordsByTfidf(self,regexps_list,num_keywords_to_choose_array):
#         # Inputs:
#         #    num_keywords_to_choose_array: if positive then function will return that many keywords, if negative value L then return len(regexps_list)/|L| keywords
#     
#         
#         # allocate some memory for keywords
#         keywords = [[]]*len(regexps_list)
#         
#         # define stemmer
#         stemmer=PorterStemmer()
#         
#         # get feature names (i.e. potential keywords) and assign idf to a dictionary
#         feature_names=self.tfidf_vectorizer.get_feature_names()
#         idf=self.tfidf_vectorizer.idf_-1
#         idf_dict={feature_names[x]:idf[x] for x in range(len(feature_names))}
#         
#         for doc_num in range(len(regexps_list)):
#             # calculate term frequencies
#             c=Counter(regexps_list[doc_num])
#             term_list=c.most_common()
#             # tf_dict={' '.join([stemmer.stem(w) for w in nltk.word_tokenize(term_list[x][0]) if not w in stopwords.words('english')]): term_list[x][1] for x in range(len(term_list))}
#             tf_dict={' '.join([stemmer.stem(w) for w in nltk.word_tokenize(term_list[x][0]) if not w in stopwords.words('english')]): term_list[x][1] for x in range(len(term_list)) if term_list[x][0] in idf_dict}
#         
#             # now calculate tf-idf
#             tfidf_dict={key: idf_dict[key]*tf_dict[key] for key in tf_dict.keys()}
#             if num_keywords_to_choose_array[doc_num] >= 0:
#                 num_keywords_to_choose=num_keywords_to_choose_array[doc_num]
#             else:
#                 num_keywords_to_choose=int(max(len(tfidf_dict)/abs(num_keywords_to_choose_array[doc_num]),1))
#             sorted_tfidf_keys=sorted(tfidf_dict.iteritems(),key=operator.itemgetter(1),reverse=True)[0:num_keywords_to_choose]
#             
#             keywords[doc_num]=[sorted_tfidf_keys[x][0] for x in range(len(sorted_tfidf_keys))]
#             
#         return keywords
    
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


    def ne(self,tagged):
        ne_tree = nltk.chunk.ne_chunk(tagged)
        ne_list = self.traverse_ne(ne_tree, ne_type="PERSON",ne_list=[]) + self.traverse_ne(ne_tree, ne_type="ORGANIZATION",ne_list=[])
        ne_list = list(set(ne_list)) # remove duplicates
        ne_list = [ne_list[x].lower() for x in range(len(ne_list))]
        return ne_list
    
    # identify noun phrases from POS-tagged text
    def nps(self,tagged):
        np_grammar = r"""
             NP: {<DT|PP\$>?<JJ>*<NN.*>+}   # chunk determiner/possessive, adjectives and noun/s
            """
        pnp_grammar  = r"""
             NP: {<NNP>*}   # series of proper nouns"""

        cp=nltk.RegexpParser(pnp_grammar)
        chunked_text=cp.parse(tagged)
        np_list=self.traverse_np(chunked_text,np_list=[])
        np_list_lower=[np_list[x].lower() for x in range(len(np_list))]
        return np_list_lower
    
    def scoreNeNp(self,ne_list,np_list,keyword_list):
        
        #turn keyword_list into dict
        keyword_dict={key: value for (key,value) in keyword_list}
        
        # if a noun phrase has a named entity in it, set the whole noun phrase as a named entity.
        ne_phrase_list=[[np_list[x] for x in range(len(np_list)) if ne_list[y] in np_list[x]] for y in range(len(ne_list))]
        ne_phrase_list=list(set(itertools.chain(*ne_phrase_list)))
        
        # for each NE phrase, assign it a score equal to the highest TextRank score among all words in the phrase
        tr_scores = [keyword_dict[key] for key in ne_phrase_list[0].split(' ') if key in keyword_dict.keys()]
        
        # change spaces in ne phrases to underscores so they can be made into keys
        # kwp_score = keyword_phrase score
        kwp_score=[]
        for ne_phrase_num in range(len(ne_phrase_list)):
            score_list=[keyword_dict[key] for key in ne_phrase_list[ne_phrase_num].split(' ') if key in keyword_dict.keys()]
            if len(score_list)==0: score_list=[0]
            kwp_score.append(np.max(np.array(score_list)))
        
        # remove phrases that have Textrank score = 0, since they are not keywordy enough to keep
        kwp_score_nonzero=[(ne_phrase_list[x].replace(' ','_'),kwp_score[x]) for x in range(len(kwp_score)) if kwp_score[x]>0]
        
        
        return kwp_score_nonzero
    
    def intersectAndCleanNeNp(self,ne_list,np_list,keyword_list):
        ## determine which keywords are named entities by taking the intersection (AND) of the keyword and named entity list
        # ne_keyword_list=[ne_list[x] for x in range(len(ne_list)) if ne_list[x].lower() in keyword_list[doc_num]] 
        ne_keyword_list=[ne_list[x] for x in range(len(ne_list)) if True in [ne_list[x].lower() in keyword_list[y] for y in range(len(keyword_list))]] 
            
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
        return ne_list_no_dup

    def load(self):
        
        # load tf-idf representation
        with open('./data/tfidf_vectorizer.p', 'r') as f:
            self.tfidf_vectorizer=pickle.load(f)
        with open('./data/tfidf.p', 'r') as f:
            self.tfidf=pickle.load(f)
        
        # load tf representation
        with open('./data/tf_vectorizer.p', 'r') as f:
            self.tf_vectorizer=pickle.load(f)
        with open('./data/tf.p', 'r') as f:
            self.tf=pickle.load(f)
        #self.feature_names=self.tf_vectorizer.get_feature_names()
        
        # load LDA model
        with open('./data/lda.p', 'r') as f:
            self.lda=pickle.load(f)
        
        # load keywords and named entities
        #loc2='/home/ryan/Documents/FBT_corpus/processed-11-11-17.json'
        loc1='./data/all_reports_processed2.json'
        self.processed_df = pd.read_json(loc1)
        
        print "Corpus has %s documents" % len(self.processed_df)
  
    ## WhosWho? function.  Uses keyword extraction, named entity recognition, and tfidf information retrieval to 
    ## Identify prominent individuals relevant to a user's query    
    def ww2(self,query):
        
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
        cosine_similarities = linear_kernel(self.tfidf, response).flatten() #flatten collapses into one dimension
        sorted_cs_arg=cosine_similarities.argsort()[::-1]
        sorted_cosine_similarities=cosine_similarities[sorted_cs_arg]
        top_docs_indicies=[sorted_cs_arg[x] for x in range(len(sorted_cosine_similarities)) if sorted_cosine_similarities[x]>sorted_cosine_similarities[0]/10]
        

                
        doc_nums=self.processed_df['doc_num']
        top_docs_bool=pd.Series([bool(np.where(doc_nums[ii] in top_docs_indicies,True,False)) for ii in range(len(doc_nums))])
        top_docs=self.processed_df.loc[top_docs_bool]
        
        #################################################################
        # sanity check.  Make sure the row numbers of the tfidf table match the 'doc_num' index in top_docs.  If sanity_check1 is false, then something is wrong
        # sanity_check1=np.sum(np.array([int(top_docs.iloc[x]['doc_num']) in top_docs_indicies for x in range(len(top_docs_indicies))]))==len(top_docs_indicies)
        
        # access by boolean - accessing by integer index gives wrong documents!
        # I verified that this is accessing the right documents by comparing the following
        # We need to call self.getText() to run some of these sanity checks
        # 1) top_docs.iloc[0]['FBT_keywords'] == self.text_df.iloc[top_docs.iloc[0]['doc_num']]['keywords']  !!! true
        # 2) top_docs_indicies[0] == top_docs.iloc[0]['doc_num'] !!! false (expected)
        # 3) top_docs[top_docs['doc_num']==top_docs_indicies[0]]['doc_num'] == top_docs_indicies[0] !!! true
        # 4) self.text_df.iloc[top_doc_indicies[0]]['FBT_Keywords']
        # 5) self.text_df.loc[1,'keywords'] == self.processed_df[self.processed_df['doc_num']==1]['FBT_keywords'] !!! true
        
        # now put all of the keywords, scores in to a dictionary with year tags
        top_docs_nonempty = top_docs[top_docs['year']>0]
        years=[str(year) for year in set(top_docs_nonempty['year'])]
        kw_score_dict={}
        
        # generate list of keywords
        keyword_list=list(top_docs_nonempty['TR_keywords'])
        keyword_list=[keyword_list[x].replace('_',' ').split(',') for x in range(len(keyword_list))]
        keyword_list_all=list(itertools.chain.from_iterable(keyword_list))
        
        #make list of keyword scores
        keyword_scores_list=list(top_docs_nonempty['TR_keyword_scores'])
        keyword_scores_list_split=[keyword_scores_list[x].replace('_',' ').split(',') for x in range(len(keyword_scores_list))]
        keyword_scores_list_split=list(itertools.chain.from_iterable(keyword_scores_list_split))
        #keyword_scores_list=[keyword_scores_list[i] if bool(keyword_scores_list[i]) else u'0' for i in range(len(keyword_scores_list))]
        keyword_scores_array=np.array([float(keyword_scores_list_split[i]) if bool(keyword_scores_list_split[i]) else 0. for i in range(len(keyword_scores_list_split))])
           
        # generate array of years    
        year_list=list(itertools.chain.from_iterable([[float(month)/12+float(year)]*len(keyword_list[doc_index]) for (month,year,doc_index) in zip(list(top_docs_nonempty['month']),list(top_docs_nonempty['year']),range(len(top_docs_nonempty)))]))
        year_array=np.round(np.array(year_list),2)
        
        # generate list of tf-idf scores
        all_doc_indicies=list(top_docs_nonempty['doc_num'])
        doc_score_list=[[cosine_similarities[all_doc_indicies[doc_index]]]*len(keyword_list[doc_index]) for doc_index in range(len(all_doc_indicies))]
        doc_tfidf_score_array=np.array(list(itertools.chain.from_iterable(doc_score_list)))
        
        # double check later, but I probably don't need to loop through the years like this
#         for year in years:
#             docs_year=top_docs[top_docs['year']==int(year)]
#             keyword_list_all=[]
#             keyword_scores_list_all=[]
#             
#             
#             
#             # make list of keywords
#             keyword_list=list(docs_year['TR_keywords'])
#             keyword_list=[keyword_list[x].replace('_',' ').split(',') for x in range(len(keyword_list))]
#             keyword_list_all.extend(list(itertools.chain.from_iterable(keyword_list)))
#             
#             #make list of keyword scores
#             keyword_scores_list=list(docs_year['TR_keyword_scores'])
#             keyword_scores_list_split=[keyword_scores_list[x].replace('_',' ').split(',') for x in range(len(keyword_scores_list))]
#             #keyword_scores_list_all_unicode=list(itertools.chain.from_iterable(keyword_scores_list_split))
#             keyword_scores_list_all=[float(score) if len(score)>0 else 0 for score in list(itertools.chain.from_iterable(keyword_scores_list_split))]
#             #keyword_scores_list_all.extend([float(score) for score in list(itertools.chain.from_iterable(keyword_scores_list))])
#             
#             doc_score_list_all=[]
#             this_year_doc_indicies=list(docs_year['doc_num'])
#             for doc_num in range(len(docs_year)):
#                 doc_score_list_all.extend([cosine_similarities[this_year_doc_indicies[doc_num]]]*len(keyword_list[doc_num]))
#                 #make list of keyword scores 
#             
#             month_list = list(docs_year['month'])
#             year_list.extend(np.array([float(month) for month in month_list])/12.+float(year))
              
        print 'Found ' + str(len(top_docs)) + ' documents relevant to query "' + query + '"' 
        #self.search_results=zip(year_list,keyword_list_all,keyword_scores_array,doc_tfidf_score_array)
        self.search_results={'year': year_list, 'keywords': keyword_list_all,'TR_score':keyword_scores_array,'tfidf_score':doc_tfidf_score_array}
                  

    def formatSearchResults(self,format='tfidf_tf_product'):
        
        if format == 'tfidf_tf_product':
            
            # make a list of years that each keyword was cited
            data=self.search_results

            #make a list of scores 100*(tfidf score)*(textrank score)/max for each keyword
            data['total_score']=100*np.multiply(data['TR_score'],data['tfidf_score'])/(data['TR_score'].max()*data['tfidf_score'].max())

        return data
        
if __name__ == '__main__':
    print 'executed main'   

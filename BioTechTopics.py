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
#import matplotlib.pyplot as plt
import pickle
from BTT_functions import *

import imp

# for some reason, Bokeh can't find wordcloud module
#f, filename, description = imp.find_module('wordcloud')
#wordcloud = imp.load_module('wordcloud', None, '/home/ryan/Dropbox/Code/Virtual-Environments/ds_py27_env/lib/python2.7/site-packages/wordcloud', ('', '', 5))
#from wordcloud import WordCloud

class Topics(object):
    
    def __init__(self):
        #plt.ion()
        # self.getText()
        self.topic_titles=['Blood tests and healthcare payers','Microbiome and Bacteria','New Drugs','Clinical Drug Development','Immunotherapies',' ','Health Data',' ','Devices and Diagnostics','Topic 9',]
    
        print('\nTopics instance ready')
    # code from https://github.com/amueller/word_cloud/blob/master/examples/simple.py
    
    def getNeScoreKw(self,doc_num,show_removed=False):
        keyword_list = self.getKeywordsByTextRank({str(doc_num):-1}, [doc_num], syntactic_filter='Filt: {<NN.*>}')[0]
        tokens = nltk.word_tokenize(cleanString(self.text_df['text_body'][doc_num]))
        tagged = nltk.pos_tag(tokens)
        ne_list=self.ne(tagged) #named entities
        np_list=self.nps(tagged) #noun phrases
        ne_list_clean=self.intersectAndCleanNeNp(ne_list,np_list,keyword_list)
        scored_kw_phrases=self.scoreNeNp(ne_list_clean,np_list,keyword_list)
        scored_kw_phrases_cleaned=self.removeRedundantKwp(scored_kw_phrases,keyword_list,show_removed=show_removed) #this is good but it still removes some named entities
        return ne_list,scored_kw_phrases_cleaned
    
    def removeRedundantKwp(self,scored_kw_phrases,keyword_lists,show_removed=False):
        # make sure each kw only appears once per document
        # Process:
        # 1) Find the most common word in the entire keyword list. (using Counter)
        # 2) For that keyword, iterate through all keyword phrases with that keyword, calculating the sum of textrank scores for all words in the phrase
        keyword_dict={key: value for (key,value) in keyword_lists}
        shortened_scored_kw_phrase_list=[]
        remaining_kw_phrases=[kw_phrase[0] for kw_phrase in scored_kw_phrases]
        while bool(remaining_kw_phrases):
            counted_kw=Counter(list(itertools.chain.from_iterable([kw_phrase.split('_') for kw_phrase in remaining_kw_phrases])))
            mc_kw=counted_kw.most_common(1)[0][0] # most common keyword
            kw_phrases_with_mc_kw = [scored_kw_phrase[0] for scored_kw_phrase in scored_kw_phrases if mc_kw in scored_kw_phrase[0]]
            
            # sum up the tr score for all words within each element of the kw_phrases_with_mc_kw list
            total_tr_score_list=np.array([np.array([keyword_dict[word.lower()] for word in set(kw_phrases_with_mc_kw[phrase_num].split('_')) if word.lower() in keyword_dict.keys()]).sum() for phrase_num in range(len(kw_phrases_with_mc_kw))])
            
            # best keyword
            shortened_scored_kw_phrase_list.append(kw_phrases_with_mc_kw[total_tr_score_list.argmax()].replace('_',' ',))
            
            # remove a keyword (phrase) if any words in the keyword phrase contain the most common keyword (mc_kw)
            remaining_kw_phrases=[phrase for phrase in remaining_kw_phrases if not bool(np.sum(np.array([mc_kw in word for word in phrase.split('_')])))]
        
        # now reassign the kw score to the shortened list
        scored_kw_phrase_dict={scored_kw_phrase[0].replace('_',' '):scored_kw_phrase[1] for scored_kw_phrase in scored_kw_phrases if scored_kw_phrase[0].replace('_',' ') in shortened_scored_kw_phrase_list}
        
        if show_removed:
            all_phrases=[kw_phrase[0] for kw_phrase in scored_kw_phrases]
            removed = [phrase.replace('_',' ') for phrase in all_phrases if phrase.replace('_',' ') not in shortened_scored_kw_phrase_list]
            print("Removed: ", removed)
            print("Not Removed: ", shortened_scored_kw_phrase_list)
            
        return scored_kw_phrase_dict

    def processCorpus(self,save_file_loc='./data/all_data_processed.json'):
        
        # basic progress bar
        df_len=len(self.text_df)
        num_dashes=100
        docs_per_dash=float(df_len)/num_dashes
        dash_num=0
        print("Progress:\n|" + "-"*num_dashes + "|")
        sys.stdout.write("|")
        
        # figure out number of docs
        if len(self.text_df.shape)==1:
            num_docs=1
        else:
            num_docs=len(self.text_df)
        
        # allocate some memory for new data frame
        new_df= pd.DataFrame({'TR_keywords':['']*df_len,'TR_keyword_scores':['']*df_len,'year':['']*df_len,'month':['']*df_len,'NE':['']*df_len,'FBT_keywords':['']*df_len,'doc_num':['']*df_len,'author':['']*df_len,'title':['']*df_len,'abs_url':['']*df_len})
        for doc_num in range(num_docs):
            # progress bar:
            if doc_num>docs_per_dash*dash_num:
                sys.stdout.write("-")
                sys.stdout.flush()
                dash_num+=1
            # do the ne, kw, np extraction
            doc_is_not_empty=False
            if num_docs>1: #if there is more than one row
                doc_is_not_empty=len(self.text_df.iloc[doc_num]['text_body'])>1
            else:
                doc_is_not_empty=bool(len(self.text_df['text_body']))
            try:
                if doc_is_not_empty: # if the document is not empty
                    ne_list,scored_kw_phrases=self.getNeScoreKw(doc_num)
                    # now assign the keywords to the dataframe columns
                    #new_df['TR_keywords'][doc_num]=','.join([scored_kw_phrases[x][0] for x in range(len(scored_kw_phrases))])
                    #new_df['TR_keyword_scores'][doc_num]=','.join([str(scored_kw_phrases[x][1]) for x in range(len(scored_kw_phrases))])
                    keys=scored_kw_phrases.keys()
                    new_df['TR_keywords'][doc_num]=','.join(keys)
                    new_df['TR_keyword_scores'][doc_num]=','.join([str(scored_kw_phrases[key]) for key in keys])
                    new_df['NE'][doc_num]=','.join(ne_list)
                    new_df['FBT_keywords'][doc_num]=','.join(self.text_df['keywords'][doc_num]).replace(' ','_')
                    new_df['doc_num'][doc_num]=doc_num
                    new_df['month'][doc_num]=str(self.text_df['date'][doc_num])[5:7]
                    new_df['author'][doc_num]=self.text_df['author'][doc_num]
                    new_df['year'][doc_num]=float(str(self.text_df['date'][doc_num])[0:4])+float(str(self.text_df['date'][doc_num])[5:7])/12.+float(str(self.text_df['date'][doc_num])[8:10])/365
                    new_df['title'][doc_num]=self.text_df['title'][doc_num]
                    new_df['abs_url'][doc_num]=self.text_df['abs_url'][doc_num]
                else:
                    new_df['TR_keywords'][doc_num]=''
                    new_df['TR_keyword_scores'][doc_num]=''
                    new_df['NE'][doc_num]=''
                    new_df['year'][doc_num]=0
                    new_df['FBT_keywords'][doc_num]=''
                    new_df['doc_num'][doc_num]=doc_num
                    new_df['month'][doc_num]=''
                    new_df['author'][doc_num]=''
                    new_df['title'][doc_num]=''
                    new_df['abs_url'][doc_num]=''
                    print("Error at document number " + str(doc_num))
                    print("\nProgress:\n|" + "-"*num_dashes + "|")
                    sys.stdout.write("|")
                    sys.stdout.write("-"*dash_num)
            except:
                print('Error on text:')
                print(self.text_df.iloc[doc_num]['text_body'])
        sys.stdout.write("|\n")
        new_df.to_json(save_file_loc)
        
#     def showTopicWordCloud(self,topic_number,fs=(6,4)):
#         
#         if topic_number<0:
#             topic_number=0
#         elif topic_number>self.lda.n_topics-1:
#             topic_number=self.lda.n_topics-1       
#         
#         # Generate a word cloud image
#         wordcloud = WordCloud(max_words=self.n_top_words).generate(self.topic_words_text[topic_number])
#         
#         # Display the image:
#         plt.figure(figsize=fs)
#         plt.imshow(wordcloud, interpolation='bilinear')
#         plt.axis("off")
#         plt.title(self.topic_titles[topic_number])
#         plt.show()
         
        
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
    
    def getText(self,json_file_loc='./data/all_reports.json',num_files=0,file_nums=0):
        # parse all of the JSON objects in the file.
        if not bool(num_files) and not bool(file_nums):
            self.text_df = pd.read_json(json_file_loc)
        elif bool(num_files):
            self.text_df = pd.read_json(json_file_loc).iloc[0:(num_files-1)]
        else:
            self.text_df = pd.read_json(json_file_loc).iloc[file_nums]
        print('Corpus contains ' + str(self.text_df.shape[0]) + ' unique files')
    
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
                    
        tfidf_vectorizer = TfidfVectorizer(tokenizer=self.tokenizeAndStemStrings, stop_words='english',ngram_range=(1,4), use_idf=True, smooth_idf = True, norm=None)
        tfidf = tfidf_vectorizer.fit_transform(self.text_df['text_body'].apply(cleanString))
            
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
            #kw=self.summaKeywords(text_filtered,self.text_df['text_body'][doc_index].encode('ascii','ignore'), ratio=0.2, words=None, language="english", split=False, scores=True)
            kw=self.summaKeywords(text_filtered,self.text_df['text_body'][doc_index], ratio=0.2, words=None, language="english", split=False, scores=True)
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
        # note that text_original.split() yields byte-typoe objects (b'xxx') which results in the problem
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
    
#     def showWhosWhoWordCloud(self,ww_list,query): 
#         # Generate a word cloud image
#         ww_text=" ".join(ww_list)
#         wordcloud = WordCloud(max_words=100).generate(ww_text)
#         
#         # Display the generated image:
#         # the matplotlib way:
#         plt.imshow(wordcloud, interpolation='bilinear')
#         plt.axis("off")
#         plt.title("Who's who for query \"" + query + "\"")
#         plt.show()
#         print("Who's who:" + " ,".join(ww_list[0:15]))


    def ne(self,tagged):
        ne_tree = nltk.chunk.ne_chunk(tagged)
        ne_list = self.traverse_ne(ne_tree, ne_type="PERSON",ne_list=[]) + self.traverse_ne(ne_tree, ne_type="ORGANIZATION",ne_list=[])
        ne_list = list(set(ne_list)) # remove duplicates
        #ne_list = [ne_list[x].lower() for x in range(len(ne_list))]
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
        #np_list_lower=[np_list[x].lower() for x in range(len(np_list))]
        return np_list
    
    def scoreNeNp(self,ne_list,np_list,keyword_list):
        
        #turn keyword_list into dict
        keyword_dict={key: value for (key,value) in keyword_list}
        
        # if a noun phrase has a named entity in it, set the whole noun phrase as a named entity.
        ne_phrase_list=[[np_list[x] for x in range(len(np_list)) if ne_list[y] in np_list[x]] for y in range(len(ne_list))]
        ne_phrase_list=list(set(itertools.chain(*ne_phrase_list)))
        
        # for each NE phrase, assign it a score equal to the highest TextRank score among all words in the phrase
        # tr_scores = [[keyword_dict[key] for key in [s.lower() for s in ne_phrase_list[ne_num].split(' ')] if key in keyword_dict.keys()] for ne_num in range(len(ne_phrase_list))]
        
        # change spaces in ne phrases to underscores so they can be made into keys
        # kwp_score = keyword_phrase score
        kwp_score=[]
        for ne_phrase_num in range(len(ne_phrase_list)):
            score_list=[keyword_dict[key] for key in [word.lower() for word in ne_phrase_list[ne_phrase_num].split(' ')] if key in keyword_dict.keys()]
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

    def load(self,data_loc='../data/'):
        with open(data_loc+'tfidf_vectorizer.p', 'rb') as f:
        #with open('./data/tfidf_vectorizer0199_nf100.p', 'r') as f:
            self.tfidf_vectorizer=pickle.load(f)
        
        # load tf-idf representation
        with open(data_loc+'tfidf.p', 'rb') as f:
        #with open('./data/tfidf0199_nf100.p', 'r') as f:
            self.tfidf=pickle.load(f)
        #with open('./data/tfidf.p', 'r') as f:
        #    self.tfidf=pickle.load(f)    
        
        # load tf representation
        #with open('./data/tf_vectorizer.p', 'r') as f:
        #    self.tf_vectorizer=pickle.load(f)
        #with open('./data/tf.p', 'r') as f:
        #    self.tf=pickle.load(f)
        #self.feature_names=self.tf_vectorizer.get_feature_names()
        
        # load LDA model
        #with open('./data/lda.p', 'r') as f:
        #    self.lda=pickle.load(f)
        
        # load keywords and named entities
        #loc2='./data/all_reports_processed.json'  #this is full corpus
        loc1='all_data_processed.json'
        self.processed_df = pd.read_json(data_loc+loc1)
        
        
        print("Corpus has %s documents" % len(self.processed_df))
  
    ## WhosWho? function.  Uses keyword extraction, named entity recognition, and tfidf information retrieval to 
    ## Identify prominent individuals relevant to a user's query    
    def ww2(self,query):
        
        ## set default ascii encoding
        #reload(sys)
        #sys.setdefaultencoding('utf8')
        
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
        #years=[str(year) for year in set(top_docs_nonempty['year'])]
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
        year_list=list(itertools.chain.from_iterable([[float(year)]*len(keyword_list[doc_index]) for (year,doc_index) in zip(list(top_docs_nonempty['year']),range(len(top_docs_nonempty)))]))
        year_array=np.round(np.array(year_list),2)

        # generate list of urls
        url_list=np.array(list(itertools.chain.from_iterable([[url]*len(keyword_list[doc_index]) for (url,doc_index) in zip(list(top_docs_nonempty['abs_url']),range(len(top_docs_nonempty)))])))
        
        # generate list of tf-idf scores
        all_doc_indicies=list(top_docs_nonempty['doc_num'])
        doc_score_list=[[cosine_similarities[all_doc_indicies[doc_index]]]*len(keyword_list[doc_index]) for doc_index in range(len(all_doc_indicies))]
        doc_tfidf_score_array=np.array(list(itertools.chain.from_iterable(doc_score_list)))
        
        # match each returned result with a doc index
        doc_num_list=list(itertools.chain.from_iterable([[all_doc_indicies[doc_index]]*len(keyword_list[doc_index]) for doc_index in range(len(top_docs_nonempty))]))
        
        print('Found ' + str(len(top_docs)) + ' documents relevant to query "' + query + '"') 
        #self.search_results=zip(year_list,keyword_list_all,keyword_scores_array,doc_tfidf_score_array)
        self.search_results={'year': year_array, 'keywords': np.array(keyword_list_all),'TR_score':keyword_scores_array,'tfidf_score':doc_tfidf_score_array,'doc_num':np.array(doc_num_list),'abs_url':url_list}
        self.search_results_status=0          

    def formatSearchResults(self,output_format='tfidf_tf_product',return_top_n=0):
        
        empty_data={'abs_url':np.array([]), 'total_score':np.array([]), 'year':np.array([]), 'keywords':np.array([]), 'TR_score':np.array([]), 'doc_num':np.array([]), 'tfidf_score':np.array([])}
        
        if output_format == 'tfidf_tf_product':
            #make a list of scores 100*(tfidf score)*(textrank score)/max for each keyword
            if self.search_results['TR_score'].size > 0:
                #keys_to_sort=['year','keywords','TR_score','tfidf_score','total_score']
                if self.search_results_status==0:
                    self.search_results['total_score']=100*np.multiply(self.search_results['TR_score'],self.search_results['tfidf_score'])/(self.search_results['TR_score'].max()*self.search_results['tfidf_score'].max())
                    sorted_args=np.argsort(self.search_results['total_score'])[::-1]
                    self.search_results={key:self.search_results[key][sorted_args] for key in self.search_results.keys()}
                    #self.search_results['total_score']=self.search_results['total_score'][sorted_args]
                    #self.search_results[keys_to_sort[0]]=self.search_results[keys_to_sort[0]][sorted_args]
                    #self.search_results[keys_to_sort[1]]=list(np.array(self.search_results[keys_to_sort[1]])[sorted_args])
                    #self.search_results[keys_to_sort[2]]=self.search_results[keys_to_sort[2]][sorted_args]
                    #self.search_results[keys_to_sort[3]]=self.search_results[keys_to_sort[3]][sorted_args]
                    self.search_results_status=1 #indicates that the data is already sorted
                    #data_sorted={key :data[key][sorted_args] for key in keys_to_sort}
                if bool(return_top_n):
                    #return_data={'year':[],'keywords':[],'TR_score':[],'tfidf_score':[],'total_score':[]}
                    #return_data[keys_to_sort[0]]=self.search_results[keys_to_sort[0]][:return_top_n]
                    #return_data[keys_to_sort[1]]=self.search_results[keys_to_sort[1]][:return_top_n]
                    #return_data[keys_to_sort[2]]=self.search_results[keys_to_sort[2]][:return_top_n]
                    #return_data[keys_to_sort[3]]=self.search_results[keys_to_sort[3]][:return_top_n]
                    #return_data[keys_to_sort[4]]=self.search_results[keys_to_sort[4]][:return_top_n]
                    return_data={key:self.search_results[key][:return_top_n] for key in self.search_results.keys()}
                else:
                    return_data=self.search_results
                    #data={self.search_results[key][:return_top_n] for key in keys_to_sort}
            else:
                return_data=empty_data
                self.search_results_status=-1
        if output_format == 'integrate_score':
            if self.search_results_status==1:
                years_floor=np.floor(self.search_results['year'])
                years = range(int(years_floor.min()),int(years_floor.max()+1))
                year_score=[sum(self.search_results['total_score'][years_floor==float(year)]) for year in years]
                return_data={'year':np.array(years),'year_score':np.array(year_score)}
            elif self.search_results_status==-1:
                return_data=empty_data
            else:  
                self.formatSearchResults(self,output_format='tfidf_tf_product',return_top_n=return_top_n)                         
        return return_data
        
if __name__ == '__main__':
    print('executed main')   

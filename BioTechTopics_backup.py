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

class Topics(object):
    
    def __init__(self):
        print 'Reding corpus'
        self.raw_text_list,self.text_list,self.keyword_list,self.num_keywords_list=self.getTextAndKeywords('/home/ryan/Dropbox/Code/Data/FierceBiotech')
        self.removeDuplicateFiles()
        print 'Corpus contains ' + str(len(self.text_list)) + ' unique files'
        print 'Training Count Vectorizer'
        self.tf, self.tf_vectorizer, self.feature_names=self.trainCountVectorizer(self.text_list)
        print 'Done'
        
    def performLDA(self,n_components):
        print('Start training LDA model')
        self.lda = LatentDirichletAllocation(n_components, max_iter=30, learning_method='online', learning_offset=50., random_state=0)
        self.lda.fit(self.tf)
        print('Done  training LDA model')
    
    def printTopWords(self,n_top_words): #from scikit-learn.org
        for topic_idx, topic in enumerate(self.lda.components_):
            message = "Topic #%d: " % topic_idx
            message += " ".join([self.feature_names[i]
                                 for i in topic.argsort()[:-n_top_words - 1:-1]])
            print(message)
        print()
    
    def removeDuplicateFiles(self):
        new_text_list=[]
        new_raw_text_list=[]
        new_keyword_list=[]
        new_num_keywords_list=[]
        #new_text_list=[self.text_list[0]]
        for doc_num in range(2,len(self.text_list)):
            duplicate=0
            for previous_doc_num in range(doc_num-1):
                if self.text_list[doc_num][0:30]==self.text_list[previous_doc_num][0:30]:
                    duplicate=1
                    break
            if duplicate == 0:
                new_text_list.append(self.text_list[doc_num])
                new_raw_text_list.append(self.raw_text_list[doc_num])
                new_keyword_list.append(self.keyword_list[doc_num])
        self.text_list = new_text_list
        self.keyword_list = new_keyword_list
        self.num_keywords_list = new_num_keywords_list
        self.raw_text_list=new_raw_text_list
    
    def stemKeywords(self,keyword_list):
        stemmer=PorterStemmer()
        for doc_num in range(len(keyword_list)):
            for keyword_num in range(len(keyword_list[doc_num])):
                keyword_list[doc_num][keyword_num]=' '.join([stemmer.stem(word) for word in nltk.word_tokenize(keyword_list[doc_num][keyword_num])])
                pass
        return keyword_list
    
    def get_text(self,json_file_loc):
        # parse all of the JSON objects in the file.  If there are more than one object, then
        # each object will be stored in an array
        
        # this anonymous function only says true if "text_body" element is present in structure.  Used this type of statement
        # because the left bool only exists if the right bool is true
        text_exist = lambda json_obj_list,x : bool(len(json_obj_list[x]['result']['extractorData']['data'][0]["group"][0].keys())==2) if bool(json_obj_list[x]['result']['extractorData']['data']) else False
        

        
        with open(json_file_loc) as data_file:
            text=data_file.read()
            seq_num_loc_total=0
            seq_end_list=[0]*100
            seq_num_loc=0
            num_json_objects=-1
            
            # determine number of JSON objects
            while seq_num_loc>=0:
                num_json_objects+=1
                seq_num_loc=text[seq_num_loc_total:].find("sequenceNumber")
                seq_num_loc_total+=seq_num_loc+len("sequenceNumber")
                seq_end_list[num_json_objects]=seq_num_loc_total
            
            # now make an list of json objects
            #seq_num_loc=text.find("sequenceNumber")
            json_obj_list = [{} for _ in range(num_json_objects)]#initialize list of json objects
            end_loc_previous=0
            for json_object_num in range(num_json_objects):
                
                # find the end of the JSON object
                json_object_end_loc=seq_end_list[json_object_num] + text[seq_end_list[json_object_num]:].find("}}")+4
        
                
                # add JSON object to list
                json_obj_list[json_object_num]=json.loads(text[end_loc_previous:json_object_end_loc])
                end_loc_previous=json_object_end_loc
    
        # now make a list of string object with (1) lowercase only (2) punctuation replaced with spaces (3) newline, carrage returns r (4) only include entries that are not empty
        

        # the boolean statement in the list comprehension ensures that there is a "text_body" entry in the json structure.  If there's not, then the document is skipped.
         # the text_exist anonymous function returns a boolean if the the "text_body" field is there
        raw_text_body = [json_obj_list[x]['result']['extractorData']['data'][0]['group'][0]['text_body'][0]['text'] for x in range(num_json_objects) if text_exist(json_obj_list,x)]
        text_body = [json_obj_list[x]['result']['extractorData']['data'][0]['group'][0]['text_body'][0]['text'].lower().encode('ascii','replace').replace('\n', ' ').replace('\r', ' ').replace('?','').translate(None, string.punctuation) for x in range(num_json_objects) if text_exist(json_obj_list,x)]
        # text_body = [json_obj_list[x]['result']['extractorData']['data'][0]['group'][0]['text_body'][0]['text'].lower().encode('ascii','ignore').translate(None, string.punctuation).replace('\n', ' ').replace('\r', ' ') for x in range(num_json_objects)]
        # text_body = [json_obj_list[x]['result']['extractorData']['data'][0]['group'][0]['text_body'][0]['text'].lower().encode('ascii','replace').replace('?s',' ').replace('?',' ').translate(None, string.punctuation).replace('\n', ' ').replace('\r', ' ') for x in range(num_json_objects)]
        text_keywords = [json_obj_list[x]['result']['extractorData']['data'][0]['group'][0]['keywords'][0]['text'].lower().encode('ascii','ignore').replace('\r', '').replace(', ',',').split(',') for x in range(num_json_objects) if text_exist(json_obj_list,x)]
            
        #print json_obj_list[2]['result']['extractorData']['data'][0]['group'][0]['Mktofieldwrap'][0]['text']
        #text_list=[]
        return raw_text_body,text_body,text_keywords
    
    def getTextAndKeywords(self,parent_directory):
        text_list = []
        raw_text_list=[]
        keyword_list = []
        num_keywords_list=[]
        file_id_list=[]
        
        # get directory of corpus.  This code assumes that corpus is in the WhosWhoBioTech folder
        corpus_directory='/home/ryan/Dropbox/Code/WhosWhoBioTech/FierceBiotech'
        
        for subdir, _, files in os.walk(corpus_directory):
            for file_name in files:
                file_path = subdir + os.path.sep + file_name
                raw_text_doc_list,text_doc_list,keyword_doc_list=self.get_text(file_path)
                #stemmed=[tokenizeAndStemStrings(text_list[x]) for x in range(len(text_list))]
                for doc_num in range(len(text_doc_list)):
                    file_id_list.append(file_name + "##" + str(doc_num))
                    text_list.append(text_doc_list[doc_num])
                    raw_text_list.append(raw_text_doc_list[doc_num])
                    keyword_list.append(keyword_doc_list[doc_num])
                    num_keywords_list.append(len(keyword_doc_list[doc_num]))
                    
        return raw_text_list,text_list,keyword_list,num_keywords_list
    
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
    
    def trainCountVectorizer(self,text_list):
                    
        tf_vectorizer = CountVectorizer(tokenizer=self.tokenizeAndStemStrings, stop_words='english',ngram_range=(1,1))        
        tf = tf_vectorizer.fit_transform(text_list)
        feature_names=tf_vectorizer.get_feature_names()
    
        return tf, tf_vectorizer, feature_names
    
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


    
    def getKeywordsByTfidf(self,regexps_list,num_keywords_to_choose_array):
        # Inputs:
        #    num_keywords_to_choose_array: if positive then function will return that many keywords, if negative value L then return len(regexps_list)/|L| keywords
    
        
        # allocate some memory for keywords
        keywords = [[]]*len(regexps_list)
        
        # define stemmer
        stemmer=PorterStemmer()
        
        # get feature names (i.e. potential keywords) and assign idf to a dictionary
        feature_names=self.tfidf.get_feature_names()
        idf=self.tfidf.idf_-1
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
        response=self.tfidf.transform([query])
        cosine_similarities = linear_kernel(self.tfs, response).flatten() #here, tfs has the actual tfidf matrix
        sorted_cs_arg=cosine_similarities.argsort()[::-1]
        sorted_cosine_similarities=cosine_similarities[sorted_cs_arg]
        #top_docs_indicies=sorted_cosine_similarities[:-11:-1]
        top_docs_indicies=[sorted_cs_arg[x] for x in range(3) if sorted_cosine_similarities[x]>sorted_cosine_similarities[0]/10]
        top_docs=[self.text_list[x] for x in top_docs_indicies]
        print 'Found ' + str(len(top_docs)) + ' documents relevant to query "' + query + '"'

        ## extract keywords from the top 10 matches from the tfidf query
        if kw_algorithm=='textrank':
            # find the top 10 keywords of each document by textrank
            keyword_list = self.getKeywordsByTextRank(top_docs, [20]*len(top_docs_indicies), 'Filt: ' + self.grammar_textrank)
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
            tokens = nltk.word_tokenize(self.raw_text_list[top_docs_indicies[doc_num]].encode('utf-8','ignore'))
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
        
        whos_who=list(set(whos_who))
        if not bool(whos_who):
            print('Unfortunately there are no key people in the corpus matching the query '+query)
        else:
            print 'Who''s who for the query "' + query + '":'
            print(whos_who)
        return whos_who            

if __name__ == '__main__':
    print 'executed main'   

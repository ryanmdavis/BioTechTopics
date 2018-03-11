import BioTechTopics as BTT
t=BTT.Topics() #about 2 minutes
t.load(data_loc='./data/')
t.ww2('cancer')
data_scatter_dict = t.formatSearchResults(output_format='tfidf_tf_product',return_top_n=200)
print('test')
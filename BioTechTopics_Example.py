from BioTechTopics import Topics
t=Topics()
t.getText()
t.processCorpus()

"""
# cases where NE were removed: 2503 (Miles White), 2500 (Maurice Greenberg)
for x in range(2500,2509):
    ne_list,scored_kw_phrases=t.getNeScoreKw(x,show_removed=True)
    print '\n'
"""
from BioTechTopics import Topics
t=Topics() #about 2 minutes
t.performLDA(20)
t.printTopWords(20)
t.showWordCloud(30)
print"test"
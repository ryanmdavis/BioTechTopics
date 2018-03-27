from BioTechTopics import Topics
t=Topics()
t.getText(json_file_loc='/home/ryan/Dropbox/Code/DataIncubatorChallenge/BioTechTopics/data/all_reports.json',num_files=10)
t.processCorpus(save_file_loc='../data/all_data_processed_temp.json')
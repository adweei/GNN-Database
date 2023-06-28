import json
import pandas as pd
import numpy as np

id = open('D:/GNN/MyResearch-main/MyResearch-main/ElonMusk/2023-02-16/Graph/encoding_table.json')
user_id = json.load(id) 
id.close()
target_tweet_id = '1625695877326340102'
all_tweet_retweet_time = pd.read_excel('D:/GNN/MyResearch-main/MyResearch-main/ElonMusk/2023-02-16/user_tweets/RetweetTime_Distribution.xlsx')
all_user_retweet_time_distribution = np.zeros((len(user_id),5))
print(all_tweet_retweet_time.index[all_tweet_retweet_time['referenced'].astype(str) ==  target_tweet_id].tolist())
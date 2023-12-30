import json
import pandas as pd
import numpy as np
import torch
import math
import os
import copy
import dgl
import re
import torch as th
from dgl.data.utils import load_graphs
from dgl.data.utils import save_graphs
# import torch.nn.functional as F 
from scipy import stats

elon_mask_base_graph_data_dir = 'D:/GCN_Twitter/ElonMusk/2023-02-16/'          #D:/GNN/new_data/MyResearch-main/MyResearch-main/ElonMusk/2023-05-01/
elon_mask_test_graph_data_dir = 'D:/GCN_Twitter/ElonMusk/2023-02-20/'          #D:/GNN/new_data/MyResearch-main/MyResearch-main/ElonMusk/2023-05-06/

tweet_retweet_relation = elon_mask_base_graph_data_dir + 'retweeters/'
# 計算每位target user retweet幾篇
id = open(elon_mask_base_graph_data_dir + 'Graph/encoding_table.json')                                        #把userid給讀入
user_id = json.load(id)
id.close()
temp_user_id_list = list(user_id.keys())
user_id_list = [int(x) for x in temp_user_id_list]
user_id_set = set(user_id_list)
# print(type(user_id_set))

all_user_retweet_file = []                                                                                 #將所有retweet紀錄檔案的檔名紀錄
for path in os.listdir(tweet_retweet_relation):
    if os.path.isfile(os.path.join(tweet_retweet_relation, path)):
        all_user_retweet_file.append(path)

id_count_list = [0] * len(user_id_list)
# print(all_user_retweet_file)
count = 0
for retweet_file in all_user_retweet_file:
    retweet = open(tweet_retweet_relation + retweet_file)                                                   #把user 的follow情況給讀入
    retweet_user_list = json.load(retweet)
    retweet.close()
    for tweet in retweet_user_list:  
        retweet_user_set = set(retweet_user_list[tweet])
        # print('retweet_user_set: ', len(retweet_user_set))
        # print('user_id_set: ', user_id_set)
        intersection_result = retweet_user_set.intersection(user_id_set)
        # print('intersection_count: ', len(intersection_result))
        for i in range(len(user_id_list)):
            if user_id_list[i] in intersection_result:
                id_count_list[i] += 1                                                                 
# print(id_count_list)
data = {
  "id": temp_user_id_list,
  "user_retweet_count": id_count_list
}
df = pd.DataFrame(data)
with pd.ExcelWriter(elon_mask_base_graph_data_dir + 'user_retweet_count.xlsx') as writer:
    df.to_excel(writer,index=False)
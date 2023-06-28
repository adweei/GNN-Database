import numpy as np
import json
import matplotlib.pyplot as plt
import seaborn as sns
import re
from dgl.data.utils import load_graphs
import os

# a = np.load('D:/GNN/MyResearch-main/MyResearch-main/ElonMusk/2023-02-16/label/label_for_0.npy')
# print(len(np.where(a == 1)[0]))
# print(len(np.where(a == 0)[0]))
# feature = np.load('D:/GNN/MyResearch-main/MyResearch-main/ElonMusk/2023-02-20/retweet_distribution/0_graph_retweet_distribution.npy')
# print(len(feature))

for nubmer in range(48):
    feature = np.load('D:/GNN/new_data/MyResearch-main/MyResearch-main/ElonMusk/2023-05-01/retweet_distribution/'+str(nubmer)+'_graph_retweet_distribution.npy')
    # print('號碼 {} 的graph 有 {} 的轉推數'.format(nubmer, len(np.where(feature[:,9] == 1)[0])))
    if len(np.where(feature[:,9] == 1)[0]) < 1000 :
        print('號碼 {} 的graph 有 {} 的轉推數'.format(nubmer, len(np.where(feature[:,9] == 1)[0])))

# f = open('D:/GNN/MyResearch-main/MyResearch-main/ElonMusk/2023-02-20/data_graph/graph_tag.txt')
# text = []
# for line in f:
#     content = re.split(": |\\n",line)
#     text.append(content[:2])
# print(text)

# dataset,label = load_graphs("D:/GNN/MyResearch-main/MyResearch-main/ElonMusk/2023-02-16/Graph/base_graph.bin")
# Graph = dataset[0]
# print(Graph.number_of_nodes())
# print(Graph.number_of_edges())
# print(np.max(np.array(Graph.edata['weight'])))
# print(np.min(np.array(Graph.edata['weight'])))

# f2 = open('D:/GNN/MyResearch-main/MyResearch-main/ElonMusk/2023-02-16/hashtags/selected_hashtag.json')
# user_id = json.load(f2) 
# f2.close()
# print(len(user_id))

# graph_list = list()
# for it in os.scandir('D:/GNN/MyResearch-main/MyResearch-main/ElonMusk/2023-02-20/data_graph'):    #需要建立幾個model
#     if it.is_dir():
#         graph_list.append(it.path+"/")

# print(graph_list)
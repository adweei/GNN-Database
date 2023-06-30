from dgl.data.utils import load_graphs
from dgl.data.utils import save_graphs
import torch as th
import pandas as pd
import numpy as np
import os
import dgl
import json

time_graph_dir = 'D:/GCN_Twitter/ElonMusk/2023-02-16/'
all_base_graph_dir = list()
id = open(time_graph_dir + 'Graph/encoding_table.json')           #把userid給讀入
user_id = json.load(id)
id.close()
for it in os.scandir(time_graph_dir + 'base_graph_for_model/'):
    if it.is_dir():
        all_base_graph_dir.append(it.path + "/")
popular = pd.read_excel(time_graph_dir + 'user_profile/Profile_new.xlsx')
popular[['id','popular']] = popular[['id','popular']].astype(str)
popular = popular[['id','popular']]
dataset,labels =  load_graphs(time_graph_dir + 'Graph/base_graph.bin')
base_graph = dataset[0]
u,v = base_graph.edges()
edge_weight = base_graph.edata['weight']
src_vertx = np.array(u)
times = 0
for per_tweet in range(len(all_base_graph_dir)):#
    Save_train_Path = all_base_graph_dir[per_tweet] + 'train/'
    Save_test_Path = all_base_graph_dir[per_tweet] + 'test/'
    tweet_number = all_base_graph_dir[per_tweet].split("/")
    tweet_number = tweet_number[-2]

    label = np.load(time_graph_dir + 'label/label_for_' + str(tweet_number)+'.npy')

    print(base_graph.number_of_nodes())
    print(len(label))
    base_graph.ndata['label'] = th.from_numpy(label)
    isolate_count = 0
    for vertx in user_id.keys():
        try:
            would_be_change_edge_list = np.where(src_vertx == user_id[vertx])
            print('orgin:',edge_weight[would_be_change_edge_list])
            if(label[user_id[vertx]] == 1):
                edge_weight[would_be_change_edge_list] += 1
                print('if active:',edge_weight[would_be_change_edge_list])
                pop = popular.loc[user_id[vertx]].tolist()
                print('popular: ',(float(pop[1])+1))
                print('?')
                edge_weight[would_be_change_edge_list] *= (float(pop[1])+1)
                print('??')
                print('after popular: ',edge_weight[would_be_change_edge_list])
            edge_weight[would_be_change_edge_list] = np.round(edge_weight[would_be_change_edge_list])
            print('final: ',edge_weight[would_be_change_edge_list])
        except:
            isolate_count += 1
        break
    base_graph.edata['weight'] = edge_weight
    break
    # for times in range(5):
    #     mask = th.rand(base_graph.number_of_nodes())
    #     train_mask = mask < 0.8
    #     train_mask = train_mask.numpy()
    #     test_mask = mask >= 0.8
    #     test_mask = test_mask.numpy()
    #     if not os.path.exists(Save_train_Path):
    #         os.makedirs(Save_train_Path)
    #     if not os.path.exists(Save_test_Path):
    #         os.makedirs(Save_test_Path)
    #     np.save(Save_train_Path+str(times),train_mask)
    #     np.save(Save_test_Path+str(times),test_mask)
    # times += 1
    # save_graphs(all_base_graph_dir[per_tweet]+str(tweet_number)+".bin", [base_graph])#將graph的結果輸出   
print(times)
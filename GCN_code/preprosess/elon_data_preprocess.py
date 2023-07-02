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
hashtag_dir = elon_mask_base_graph_data_dir + 'hashtags/'
user_profile_dir = elon_mask_base_graph_data_dir + 'user_profile/'
user_activelabel_dir = elon_mask_base_graph_data_dir + 'elon_mask_base_graph_data_dirlabel/'
base_graph_dir = elon_mask_base_graph_data_dir + 'base_graph_for_model/'
predict_graph_dir = elon_mask_test_graph_data_dir + 'data_graph/'
follow_relationship = elon_mask_base_graph_data_dir + 'followers/'
tweet_retweet_relation = elon_mask_base_graph_data_dir + 'retweeters/'
test_tweet_label =  elon_mask_test_graph_data_dir + 'label/'
base_tweet_retweet_distribution = elon_mask_base_graph_data_dir + 'retweet_distribution/'
test_tweet_retweet_distribution = elon_mask_test_graph_data_dir + 'retweet_distribution/'
def PCA(data, k=2):
    X = torch.from_numpy(data)
    X_mean = torch.mean(X, 0)
    X = X - X_mean.expand_as(X)
    # SVD
    U,S,V = torch.svd(torch.t(X))
    return torch.mm(X,U[:,:k])

def np_norm(data):
    return (data-np.min(data))/(np.max(data)-np.min(data))

def self_def_norm(data,floor,ceiling):
    return ((data-np.min(data))/(np.max(data)-np.min(data))) * (ceiling - floor) - (0 - floor)


# f = open(elon_mask_base_graph_data_dir+'Graph/encoding_table.json')
# data = json.load(f)                                                              #這邊是user流水號讀入
# print(data.values())
# f.close()
#-----------------------------------------------------------------------------------------------------------------
# workbook = pd.read_excel(elon_mask_base_graph_data_dir+'ElonMusk.xlsx')

# tweet_id = dict()                                                                #這邊是處理tweet_id轉成流水號
# for row in range(0,len(workbook),1):
#     tweet_id[int(workbook.id[row])] = (row+1)

# with open(elon_mask_base_graph_data_dir+"tweet_id.json", "w") as outfile:
#     json.dump(tweet_id, outfile)
#---------------------------------------------------------------------------------------------------------------------
# Opening JSON file
# f = open(hashtag_dir+'hashtag_count.json')                                      #這邊處理被提及次數超過20次的hashtag
  
# # returns JSON object as 
# # a dictionary
# hash_count_data = json.load(f)
  
# # Iterating through the json
# selected_hashtag = list()
# for item in hash_count_data.items():
#     if item[1] >= 20:
#         selected_hashtag.append(item[0])
# # Closing file
# f.close()
# with open(hashtag_dir+"selected_hashtag.json", "w") as outfile:
#     json.dump(selected_hashtag, outfile)
#---------------------------------------------------------------------------------------------------------------------

# f = open(hashtag_dir+'hashtag_user_mention.json')
# hashtag_data = json.load(f)
# f.close()                                                                                     #這邊處理哪個hashtag被哪個user提到
# f2 = open(elon_mask_base_graph_data_dir+'Graph/encoding_table.json')
# user_id = json.load(f2) 
# f2.close()
# f3 = open(hashtag_dir+'selected_hashtag.json')
# selected_hashtag_data = json.load(f3)
# f3.close()

# hashtag_map = np.zeros((len(user_id), len(selected_hashtag_data)))                            #先建立hashTag等大的陣列
# key_list = list(hashtag_data.keys())

# count = 0
# for keys in hashtag_data:                                                                       #看現在被選到的hashtag是否有在我們所選擇的hashtag範圍內
#     if keys in selected_hashtag_data:                                                           #如果是我們需要的hashtag，那就把有提到它的人都列出來
#         for user in hashtag_data[keys]:                                                         #根據前面的結果，把陣列內對應位置的數值改成1，代表該人有提到該hashtag
#             try:
#                 hashtag_map[user_id[user]][key_list.index(keys)] = 1
#             except:
#                 count += 1
# np.save(hashtag_dir+"orgin_hashtag_map.json", hashtag_map)
#--------------------------------------------------------------------------------------------------------------------------
# hashtag_map = np.load(hashtag_dir+"orgin_hashtag_map.json.npy")
# user_feature = PCA(hashtag_map,28)                                                              #將剛剛的結果做PCA降維到28
# user_feature = user_feature.numpy()                                                      
# np.save(elon_mask_base_graph_data_dir+'user_feature', user_feature)
# #----------------------------------------------------------------------------------------------------------------------

# id = open(elon_mask_base_graph_data_dir+'Graph/encoding_table.json')
# user_id = json.load(id)                                                                          #這邊處理的是將Hashtag的pca結果(28個)和我們自訂義的4個feature做連接並且作輸出
# id.close()

# all_user_profile = pd.read_excel(elon_mask_base_graph_data_dir+'user_profile/Profile.xlsx')
# user_feature = np.load(elon_mask_base_graph_data_dir+'user_feature.npy')
# # user_feature = np.transpose(np_norm(np.transpose(user_feature)))

# selfdefine_feature = ['followers_count', 'following_count', 'tweet_count', 'verified']
# allperson_features = list()
# allperson_popular = [0.1] * len(all_user_profile.index)
# for keys in user_id:
#     profile_feature = all_user_profile.loc[all_user_profile['id'].astype(str) == keys, ['followers_count', 'following_count', 'tweet_count', 'verified']].to_numpy(dtype=int)
#     user_index = all_user_profile.index[all_user_profile['id'].astype(str) == keys].tolist()
#     allperson_features.append(profile_feature[0])
#     if(profile_feature[0][0] != 0):
#         if(profile_feature[0][0] != 1):
#             if(math.log(profile_feature[0][0],10) !=1):
#                 allperson_popular[user_index[0]] = math.log(profile_feature[0][0],10)
# all_user_profile['popular'] = allperson_popular
# all_user_profile['id'] = all_user_profile['id'].astype(str)
# allperson_features = np.transpose(allperson_features).astype(float)
# allperson_features[0] = self_def_norm(allperson_features[0], 0, 1)
# allperson_features[1] = self_def_norm(allperson_features[1], 0, 1)
# allperson_features[2] = self_def_norm(allperson_features[2], 0, 1)
# allperson_features = np.transpose(allperson_features)
# user_feature = np.concatenate((user_feature, allperson_features), axis=1)
# np.save(elon_mask_base_graph_data_dir+'user_feature', user_feature)
# all_user_profile.to_excel(elon_mask_base_graph_data_dir+'user_profile/Profile_new.xlsx',index=False)

# print(user_feature.shape)

#-----------------------------------------------------------------------------------------------------------------------------------------------

# target_tweet = pd.read_excel(elon_mask_base_graph_data_dir+'compare5.xlsx')    #這邊是把要篩選的tweetid給讀入
# thershold = 26                                                                                              #retweet >= 1500 的推文數 0~26是27篇 
# target_tweet = target_tweet.drop(index=list(range(thershold+1,len(target_tweet))))
# target_tweet['id'] = target_tweet['id'].astype(str)
# id = open(elon_mask_base_graph_data_dir+'Graph/encoding_table.json')           #把userid給讀入
# user_id = json.load(id)
# id.close()
# user_id_list = list(user_id.keys())                                                                           #取兩個set之間的交集，藉以找出重合的target user(哪些人有追蹤這篇推文)
# user_id_list = [int(x) for x in user_id_list]
# user_id_set = set(user_id_list)

# # folder path                                                                                               #將各推文轉推情形的檔名紀錄
# tweet_retweet_path = elon_mask_base_graph_data_dir+'retweeters/'
# # list to store files
# all_tweet_retweet_file = []
# # Iterate directory
# for path in os.listdir(tweet_retweet_path):
#     # check if current path is a file
#     if os.path.isfile(os.path.join(tweet_retweet_path, path)):
#         all_tweet_retweet_file.append(path)

# all_label = list()
# label_order = list()
# for file in all_tweet_retweet_file:                                                                     #在所有的json檔中做尋找
#     retweet_file = open(tweet_retweet_path+file)
#     part_tweet_retweet_file = json.load(retweet_file)
#     for per_tweet in part_tweet_retweet_file.keys():
#         inner = target_tweet['id'].isin([per_tweet])                                                    #檢查當前tweet是否在目標tweet名單內
#         if (inner.sum() != 0):                                                                          #有的話紀錄它在原本檔案中的index和有哪些人轉推
#             label_order.append(target_tweet[target_tweet['id'].isin([per_tweet])].index.tolist())
#             target_tweet_retweet = user_id_set.intersection(set(part_tweet_retweet_file[per_tweet]))
#             all_label.append(target_tweet_retweet)
            
# for number in range(0,len(all_label),1):
#     per_tweet_label = np.zeros(len(user_id))
#     for user in all_label[number]:
#         per_tweet_label[user_id[str(user)]] = 1
#     np.save(elon_mask_base_graph_data_dir+'label/label_for_'+str(label_order[number][0]), per_tweet_label)
#---------------------------------------------------------------------------------------------------------------------------------------------------------

# #這邊是建立base graph
# id = open(elon_mask_base_graph_data_dir+'Graph/encoding_table.json')                                       #把userid給讀入
# user_id = json.load(id)
# id.close()
# src_vertx = dict()

# def find_in_target(target_dict,check):
#     if(str(check) in target_dict):
#         return True
#     else:
#         return False

# all_user_follow_file = []                                                                                 #將所有follower檔案的檔名紀錄
# for path in os.listdir(follow_relationship):
#     if os.path.isfile(os.path.join(follow_relationship, path)):
#         if(path != 'track.py'):
#             all_user_follow_file.append(path)

# all_user_retweet_file = []                                                                                 #將所有retweet紀錄檔案的檔名紀錄
# for path in os.listdir(tweet_retweet_relation):
#     if os.path.isfile(os.path.join(tweet_retweet_relation, path)):
#         all_user_retweet_file.append(path)

# for follow_file in all_user_follow_file:
#     follow = open(follow_relationship+follow_file)                                                             #把user 的follow情況給讀入
#     user_follower = json.load(follow)
#     follow.close()

#     for user in user_follower:                                                                                 #把user情況轉流水號先記錄    格式是: src:
#         if (user in user_id):                                                                                  #                                    dst:
#             dst_vertx = dict()                                                                                 #                                        次數
#             for followers in user_follower[user]:
#                 if(str(followers) in user_id):
#                     dst_vertx[user_id[str(followers)]] = 1
#             src_vertx[user_id[user]] = dst_vertx

# count = 0
# for retweet_file in all_user_retweet_file:
#     retweet = open(tweet_retweet_relation+retweet_file)                                                                     #把user 的follow情況給讀入
#     user_tweet_retweet = json.load(retweet)
#     retweet.close()
#     for tweet in user_tweet_retweet:                                                                                        #根據每篇推文
#         for ever_user in user_tweet_retweet[tweet]:                                                                         #檢查底下所有的user
#             if(find_in_target(user_id,ever_user)):                                                                          #檢查是否是最後的target user
#                 if(user_id[str(ever_user)] in src_vertx):                                                                   #檢查是否在src_dst的src列表裡
#                     leaf_user = list()
#                     leaf_user = user_tweet_retweet[tweet].copy()
#                     leaf_user.remove(ever_user)                                                                             #需要檢查除了當前的user以外 也有轉推同一篇的user是否有出現在src_dst的dst列表裡
#                     for weight_change_user in leaf_user:
#                         if(find_in_target(user_id,weight_change_user)):                                                     #檢查是否是最後的target user
#                             if(user_id[str(weight_change_user)] in src_vertx[user_id[str(ever_user)]]):                     #檢查是否出現在src_dst的dst列表
#                                 # print('src:',user_id[str(ever_user)])
#                                 # print('dst:',user_id[str(weight_change_user)])
#                                 # print('dict:',src_vertx[user_id[str(ever_user)]])
#                                 count += 1
#                                 src_vertx[user_id[str(ever_user)]][user_id[str(weight_change_user)]] += 1                   #若有出現，則將次數+1


# with open(elon_mask_base_graph_data_dir+"user_relation.json", "w") as outfile:
#     json.dump(src_vertx, outfile)

#-----------------------------------------------------------------------------------------------------------------------------------------------------------------
# id = open(elon_mask_base_graph_data_dir+'Graph/encoding_table.json')                                       #把userid給讀入
# user_id = json.load(id)
# id.close()
# vertx_number = len(user_id)
# del user_id
# src_vertx = dict()
# edge = open(elon_mask_base_graph_data_dir+'user_relation.json')                                       #把所有edge資訊讀入給讀入
# all_edge_collection = json.load(edge)
# edge.close()

# src_node = np.array([])
# dst_node = np.array([])
# weight = np.array([])
# for src_vertx in all_edge_collection:                                                                   #根據前面處裡完的src-dst-weight的dict解析結果
#     for dst_vertx in all_edge_collection[src_vertx]:
#         # print('src node is: ',src_vertx)
#         # print('dst node is: ',dst_vertx)
#         # print('edge weight is: ',all_edge_collection[src_vertx][dst_vertx])
#         # print('--------------------------------------')
#         src_node = np.append(src_node,src_vertx)
#         dst_node = np.append(dst_node,dst_vertx)
#         weight = np.append(weight,int(all_edge_collection[src_vertx][dst_vertx]))                       #將個別的結果陣列儲存
# del all_edge_collection
# weight = np.reshape(weight,(weight.shape[0],1))
# np.save(elon_mask_base_graph_data_dir+'Graph/source_vertx_collection', src_node)
# np.save(elon_mask_base_graph_data_dir+'Graph/destion_vertx_collection', dst_node)
# np.save(elon_mask_base_graph_data_dir+'Graph/edge_weight_collection', weight)
#--------------------------------------------------------------------------------------------------------------------------------------------------------------------
# id = open(elon_mask_base_graph_data_dir+'Graph/encoding_table.json')                                       #把userid給讀入
# user_id = json.load(id)
# id.close()
# vertx_number = len(user_id)
# del user_id
# src_node = np.load(elon_mask_base_graph_data_dir+'Graph/source_vertx_collection.npy')
# dst_node = np.load(elon_mask_base_graph_data_dir+'Graph/destion_vertx_collection.npy')
# weight = np.load(elon_mask_base_graph_data_dir+'Graph/edge_weight_collection.npy')
# model_graph_dir = elon_mask_base_graph_data_dir+'user_feature.npy'
# vertx_feature = np.load(model_graph_dir)
# base_graph = dgl.graph(([],[]),num_nodes = vertx_number)                                                  #將前面得到的src和dst的矩陣當作資料，一個個加入到圖中
# base_graph.add_edges(src_node.astype(int),dst_node.astype(int))
# graph_base_weight = th.from_numpy(weight)
# graph_base_weight = graph_base_weight.to(th.int64)
# graph_feature = th.from_numpy(vertx_feature).to(th.float32)
# base_graph.edata['weight'] = graph_base_weight
# base_graph.ndata['feature'] = graph_feature
# save_graphs(elon_mask_base_graph_data_dir+"Graph/base_graph.bin", [base_graph])#將graph的結果輸出
#-------------------------------------------------------------------------------------------------------------------------------------------------------------------
# all_base_graph_dir = list()
# id = open(elon_mask_base_graph_data_dir+'Graph/encoding_table.json')           #把userid給讀入
# user_id = json.load(id)
# id.close()
# for it in os.scandir(elon_mask_base_graph_data_dir+'base_graph_for_model/'):    #需要建立幾個model
#     if it.is_dir():
#         all_base_graph_dir.append(it.path+"/")
# popular = pd.read_excel(user_profile_dir+'Profile_new.xlsx')
# popular[['id','popular']] = popular[['id','popular']].astype(str)
# popular = popular[['id','popular']]
# dataset,labels =  load_graphs(elon_mask_base_graph_data_dir+"Graph/base_graph.bin")     #將建好的graph讀取出來之後把label放上去 並且對應edge去檢查是否active，再去更改edge_weight
# base_graph = dataset[0]
# u,v = base_graph.edges()
# src_vertx = np.array(u)
# edge_weight = base_graph.edata['weight'].numpy() * 0.1

# for per_tweet in range(len(all_base_graph_dir)):#
#     Save_train_Path = all_base_graph_dir[per_tweet] + 'train/'
#     Save_test_Path = all_base_graph_dir[per_tweet] + 'test/'
#     tweet_number = all_base_graph_dir[per_tweet].split("/")
#     tweet_number = tweet_number[-2]

#     label = np.load(elon_mask_base_graph_data_dir+'label/label_for_'+str(tweet_number)+'.npy')
#     base_graph.ndata['label'] = th.from_numpy(label).type(th.LongTensor)
#     isolate_count = 0
#     for vertx in user_id.keys():
#         try:
#             would_be_change_edge_list = np.where(src_vertx == user_id[vertx])
#             # print('orgin:',edge_weight[would_be_change_edge_list])
#             if(label[user_id[vertx]] == 1):
#                 edge_weight[would_be_change_edge_list] += 1
#                 # print('if active:',edge_weight[would_be_change_edge_list])
#                 pop = popular.loc[user_id[vertx]].tolist()
#                 # print('popular: ',(float(pop[1])+1))
#                 for num in would_be_change_edge_list:
#                     edge_weight[num] = edge_weight[num] * (float(pop[1]) / 5)   # /5 是要壓低pop的數值
#                 # print('after popular: ',edge_weight[would_be_change_edge_list])
#                 edge_weight[would_be_change_edge_list] = np.round(edge_weight[would_be_change_edge_list])
#                 # print('final: ',edge_weight[would_be_change_edge_list])
#         except:
#             isolate_count += 1
#     # edge_weight = np_norm(edge_weight)
#     # graph_weight = th.from_numpy(edge_weight).to(th.float32)
#     # base_graph.edata['weight'] = graph_weight
#     base_graph.edata['weight'] = th.from_numpy(edge_weight).to(th.int32)
#     base_graph.ndata['self_weight'] = th.from_numpy(np.full((base_graph.number_of_nodes(),1),np.median(edge_weight,axis=0))).to(th.float32)

#     for times in range(5):#5
#         mask = th.rand(base_graph.number_of_nodes())
#         train_mask = mask < 0.8
#         train_mask = train_mask.numpy()
#         test_mask = mask >= 0.8
#         test_mask = test_mask.numpy()
#         if not os.path.exists(Save_train_Path):
#             os.makedirs(Save_train_Path)
#         if not os.path.exists(Save_test_Path):
#             os.makedirs(Save_test_Path)
#         np.save(Save_train_Path+str(times),train_mask)
#         np.save(Save_test_Path+str(times),test_mask)
#     save_graphs(all_base_graph_dir[per_tweet]+str(tweet_number)+".bin", [base_graph])#將graph的結果輸出
#-----------------------------------------------------------------------------------------------------------------------------------------------------------------
# # #整理訓練結果
# all_model_json_file_dir = list()
# for it in os.scandir(base_graph_dir):    #需要蒐集幾個model結果
#     if it.is_dir():
#         all_model_json_file_dir.append(it.path+"/")
# all_result_arrange = list()
# for every_graph in range(27):#27
#     result_arrange = {}
#     for every_batch in range(5):#5
#         # print(every_graph)
#         # print(every_batch)
#         result = open(all_model_json_file_dir[every_graph]+str(every_batch)+'_round_result_edge_weight_Adjustment.json')           #把userid給讀入
#         ever_graph_result = json.load(result)
#         result.close()
#         result_arrange[every_batch] = ever_graph_result
#     all_result_arrange.append(pd.DataFrame.from_dict(result_arrange))

# with pd.ExcelWriter(elon_mask_base_graph_data_dir+'model_result_modify_collection.xlsx') as writer:
#     for model_result in range(len(all_result_arrange)):
#         all_result_arrange[model_result].to_excel(writer, sheet_name='model_for_'+str(model_result))
#-------------------------------------------------------------------------------------------------------------------------------------------------------------------
#將retweet時間表建立並且儲存
# all_base_graph_dir = list()
# id = open(elon_mask_base_graph_data_dir+'Graph/encoding_table.json')                                        #把userid給讀入
# user_id = json.load(id)
# id.close()
# target_tweet = pd.read_excel(elon_mask_base_graph_data_dir+'user_tweets/RetweetTime_Distribution.xlsx',sheet_name='retweet time')    #這邊是把轉推時間分布讀進來
# target_tweet = target_tweet.astype(str)
# target_tweet_list = pd.read_excel(elon_mask_base_graph_data_dir+'user_tweets/RetweetTime_Distribution.xlsx',sheet_name='total count of each hr')    #這邊把預測推文的轉推數讀進來
# raw_target_tweet_list = target_tweet_list['tweet id'][list(np.where(np.array(target_tweet_list['48hr']) > 1000)[0])].values.tolist()                        #本次預測推文的原轉推人數超過 " 1000人 "
# remain_target_user = target_tweet[target_tweet['id'].isin(user_id.keys())].copy()

# all_data_graph_label = np.zeros((len(raw_target_tweet_list)+1,len(user_id),11))
# count = 0
# for target_tweet_id in raw_target_tweet_list:
#     target_tweet_retweet_time_mask = remain_target_user['referenced'].isin([str(target_tweet_id)])
#     target_tweet_retweet_time_list = list(remain_target_user['time_delta'][target_tweet_retweet_time_mask])
#     target_tweet_retweet_user = list(remain_target_user['id'][target_tweet_retweet_time_mask])
#     for number in range(len(target_tweet_retweet_user)):
#         if(int(target_tweet_retweet_time_list[number]) <= 1):
#             all_data_graph_label[count][user_id[str(target_tweet_retweet_user[number])]][0:10] = 1
#             all_data_graph_label[count][user_id[str(target_tweet_retweet_user[number])]][-1] = 1
#         elif(int(target_tweet_retweet_time_list[number]) <= 2):
#             all_data_graph_label[count][user_id[str(target_tweet_retweet_user[number])]][1:10] = 1
#             all_data_graph_label[count][user_id[str(target_tweet_retweet_user[number])]][-1] = 2
#         elif(int(target_tweet_retweet_time_list[number]) <= 4):
#             all_data_graph_label[count][user_id[str(target_tweet_retweet_user[number])]][2:10] = 1
#             all_data_graph_label[count][user_id[str(target_tweet_retweet_user[number])]][-1] = 4
#         elif(int(target_tweet_retweet_time_list[number]) <= 6):
#             all_data_graph_label[count][user_id[str(target_tweet_retweet_user[number])]][3:10] = 1
#             all_data_graph_label[count][user_id[str(target_tweet_retweet_user[number])]][-1] = 6
#         elif(int(target_tweet_retweet_time_list[number]) <= 8):
#             all_data_graph_label[count][user_id[str(target_tweet_retweet_user[number])]][4:10] = 1
#             all_data_graph_label[count][user_id[str(target_tweet_retweet_user[number])]][-1] = 8
#         elif(int(target_tweet_retweet_time_list[number]) <= 10):
#             all_data_graph_label[count][user_id[str(target_tweet_retweet_user[number])]][5:10] = 1
#             all_data_graph_label[count][user_id[str(target_tweet_retweet_user[number])]][-1] = 10
#         elif(int(target_tweet_retweet_time_list[number]) <= 12):
#             all_data_graph_label[count][user_id[str(target_tweet_retweet_user[number])]][6:10] = 1
#             all_data_graph_label[count][user_id[str(target_tweet_retweet_user[number])]][-1] = 12
#         elif(int(target_tweet_retweet_time_list[number]) <= 16):
#             all_data_graph_label[count][user_id[str(target_tweet_retweet_user[number])]][7:10] = 1
#             all_data_graph_label[count][user_id[str(target_tweet_retweet_user[number])]][-1] = 16
#         elif(int(target_tweet_retweet_time_list[number]) <= 20):
#             all_data_graph_label[count][user_id[str(target_tweet_retweet_user[number])]][8:10] = 1
#             all_data_graph_label[count][user_id[str(target_tweet_retweet_user[number])]][-1] = 20
#         elif(int(target_tweet_retweet_time_list[number]) <= 24):
#             all_data_graph_label[count][user_id[str(target_tweet_retweet_user[number])]][9:10] = 1
#             all_data_graph_label[count][user_id[str(target_tweet_retweet_user[number])]][-1] = 24
#         # print(all_data_graph_label[count][user_id[str(target_tweet_retweet_user[number])]])
#     count += 1

# if not os.path.isdir(elon_mask_base_graph_data_dir+'retweet_distribution/'):
#     os.mkdir(elon_mask_base_graph_data_dir+'retweet_distribution/')
# for graph_number in range(all_data_graph_label.shape[0]):
#     np.save(elon_mask_base_graph_data_dir+'retweet_distribution/'+str(graph_number)+'_graph_retweet_distribution_rewrite.npy',all_data_graph_label[graph_number])
#-------------------------------------------------------------------------------------------------------------------------------------------------------------------
# raw_retweet_record_file = []                                                                                 #將可能超過目標數的tweet轉推紀錄的檔名紀錄
# for path in os.listdir(base_tweet_retweet_distribution):
#     if os.path.isfile(os.path.join(base_tweet_retweet_distribution, path)):
#             raw_retweet_record_file.append(path)

# for nubmer in range(len(raw_retweet_record_file)):                                                          #計算有哪些tweet有超過標準，這邊標準為" 1200人 "
#     feature = np.load(base_tweet_retweet_distribution+str(nubmer)+'_graph_retweet_distribution.npy')
#     # print('號碼 {} 的graph 有 {} 的轉推數'.format(nubmer, len(np.where(feature[:,9] == 1)[0])))
#     if len(np.where(feature[:,9] == 1)[0]) < 1200 :
#         print('號碼 {} 的graph 有 {} 的轉推數'.format(nubmer, len(np.where(feature[:,9] == 1)[0])))
#         os.remove(base_tweet_retweet_distribution+str(nubmer)+'_graph_retweet_distribution.npy')
#------------------------------------------------------------------------------------------------------------------------------------------------------------------
# #將3/16~20的label給建立出來
# target_tweet = pd.read_excel(elon_mask_test_graph_data_dir+'compare4.xlsx')    #這邊是把要篩選的tweetid給讀入
# thershold = 28                                                                                              #retweet >= 1500 的推文數 0~28是29篇 
# target_tweet = target_tweet.drop(index=list(range(thershold+1,len(target_tweet))))
# target_tweet['id'] = target_tweet['id'].astype(str)
# id = open(elon_mask_base_graph_data_dir+'Graph/encoding_table.json')           #把userid給讀入
# user_id = json.load(id)
# id.close()
# user_id_list = list(user_id.keys())                                                                           #取兩個set之間的交集，藉以找出重合的target user(哪些人有追蹤這篇推文)
# user_id_list = [int(x) for x in user_id_list]
# user_id_set = set(user_id_list)

# # folder path                                                                                               #將各推文轉推情形的檔名紀錄
# tweet_retweet_path = elon_mask_test_graph_data_dir+'retweeters/'
# # list to store files
# all_tweet_retweet_file = []
# # Iterate directory
# for path in os.listdir(tweet_retweet_path):
#     # check if current path is a file
#     if os.path.isfile(os.path.join(tweet_retweet_path, path)):
#         all_tweet_retweet_file.append(path)

# all_label = list()
# label_order = list()
# for file in all_tweet_retweet_file:                                                                     #在所有的json檔中做尋找
#     retweet_file = open(tweet_retweet_path+file)
#     part_tweet_retweet_file = json.load(retweet_file)
#     for per_tweet in part_tweet_retweet_file.keys():
#         inner = target_tweet['id'].isin([per_tweet])                                                    #檢查當前tweet是否在目標tweet名單內
#         if (inner.sum() != 0):                                                                          #有的話紀錄它在原本檔案中的index和有哪些人轉推
#             label_order.append(target_tweet[target_tweet['id'].isin([per_tweet])].index.tolist())
#             target_tweet_retweet = user_id_set.intersection(set(part_tweet_retweet_file[per_tweet]))
#             all_label.append(target_tweet_retweet)

# for number in range(0,len(all_label),1):
#     per_tweet_label = np.zeros(len(user_id))
#     for user in all_label[number]:
#         per_tweet_label[user_id[str(user)]] = 1
#     np.save(elon_mask_test_graph_data_dir+'label/label_for_'+str(label_order[number][0]), per_tweet_label)
#-----------------------------------------------------------------------------------------------------------------------------------------------------------------
#這邊將3/16~20的data graph 給建出來
# def weight_Half_life(retweet_time,now_time):
#     return  0.5 + (1 - 0.5) * (1 / (2**(now_time - retweet_time)))                  #從轉推時間到現在作為半衰期的指數，最終趨近於0.5

# all_data_graph_dir = list()
# id = open(elon_mask_base_graph_data_dir+'Graph/encoding_table.json')           #把userid給讀入
# user_id = json.load(id)
# id.close()
# label_list = list()
# for it in os.scandir(test_tweet_retweet_distribution):    #需要建立幾個model
#     if it.is_file():
#         label_list.append(it.path+"/")
# for tweet_number in range(len(label_list)):     #創立等等要放data graph的資料夾
#     if not os.path.isdir(elon_mask_test_graph_data_dir+'data_graph/'+str(tweet_number)):
#         os.mkdir(elon_mask_test_graph_data_dir+'data_graph/'+str(tweet_number))
#     for time_point in range(10):
#         if not os.path.isdir(elon_mask_test_graph_data_dir+'data_graph/'+str(tweet_number)+'/'+str(time_point)):
#             os.mkdir(elon_mask_test_graph_data_dir+'data_graph/'+str(tweet_number)+'/'+str(time_point))
# popular = pd.read_excel(user_profile_dir+'Profile_new.xlsx')
# popular[['id','popular']] = popular[['id','popular']].astype(str)
# popular = popular[['id','popular']]
# dataset,labels =  load_graphs(elon_mask_base_graph_data_dir+"Graph/base_graph.bin")     #將建好的graph讀取出來之後把label放上去 並且對應edge去檢查是否active，再去更改edge_weight
# base_graph = dataset[0]
# u,v = base_graph.edges()
# src_vertx = np.array(u)
# edge_weight = base_graph.edata['weight'].numpy() * 0.1
# retweet_time_point_list = [1,2,4,6,8,10,12,16,20,24]

# for per_tweet in range(len(label_list)):#
#     tweet_numbers = label_list[per_tweet].split("_")
#     tweet_numbers = tweet_numbers[1].split("/")[1]
#     label = np.load(test_tweet_label+'label_for_'+str(tweet_numbers)+'.npy')
#     retweet_distribution = np.load(base_tweet_retweet_distribution+str(tweet_numbers)+'_graph_retweet_distribution_rewrite.npy')

#     for time_points in range(10):             
#         now_active = retweet_distribution[:,time_points]
#         active_retweet_time = retweet_distribution[:,10]
#         base_graph.ndata['label'] = th.from_numpy(label).type(th.LongTensor)
#         isolate_count = 0
#         for vertx in user_id.keys():
#             try:
#                 would_be_change_edge_list = np.where(src_vertx == user_id[vertx])   #首先檢查src是否在target user裡
#                 # print('orgin:',edge_weight[would_be_change_edge_list])
#                 if(now_active[user_id[vertx]] == 1):                                  #在檢查是否為active
#                     edge_weight[would_be_change_edge_list] += weight_Half_life(active_retweet_time[user_id[vertx]],retweet_time_point_list[time_points])
#                     # print('node number: ',user_id[vertx])
#                     # print('推文轉推時的時間',active_retweet_time[user_id[vertx]])
#                     # print('現在的時間',retweet_time_point_list[time_points])
#                     # print('半衰期的結果',weight_Half_life(active_retweet_time[user_id[vertx]],retweet_time_point_list[time_points]))
#                     # print('if active:',edge_weight[would_be_change_edge_list])
#                     pop = popular.loc[user_id[vertx]].tolist()                      #是active時要做 weight * pop
#                     # print('popular: ',(float(pop[1])+1))
#                     for num in would_be_change_edge_list:                           #將該user所有的out degree做 weight *pop
#                         edge_weight[num] = edge_weight[num] * (float(pop[1]) / 5)   # /5 是要壓低pop的數值
#                     # print('after popular: ',edge_weight[would_be_change_edge_list])
#                     edge_weight[would_be_change_edge_list] = np.round(edge_weight[would_be_change_edge_list]) #將結果寫回原本的weight上
#                     # print('final: ',edge_weight[would_be_change_edge_list])
#             except:
#                 isolate_count += 1
#         base_graph.edata['weight'] = th.from_numpy(edge_weight).to(th.int32)
#         base_graph.ndata['self_weight'] = th.from_numpy(np.full((base_graph.number_of_nodes(),1),np.median(edge_weight,axis=0))).to(th.float32) #每個vertx在gcn時要一起考慮自身的feature  所以給它一個weight 讓其考慮自身時可以跟周邊的edge weight類似 這邊使用中位數
#         save_graphs(elon_mask_test_graph_data_dir+'data_graph/'+str(tweet_numbers)+'/'+str(time_points)+'/'+str(tweet_numbers)+"_rewrite.bin", [base_graph])#將graph的結果輸出

#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#3/16~20的預測結果蒐集
all_predict_json_file_dir = list()
for it in os.scandir(predict_graph_dir):    #需要蒐集幾個model結果
    if it.is_dir():
        all_predict_json_file_dir.append(it.path+"/")
all_tag_result_arrange = list()
all_high_retweet_count_result_arrange = list()
all_middle_retweet_count_result_arrange = list()
all_low_retweet_count_result_arrange = list()
all_shortest_retweet_distribution_result_arrange = list()
all_shorter_retweet_distribution_result_arrange = list()
all_middle_retweet_distribution_result_arrange = list()
all_longer_retweet_distribution_result_arrange = list()
all_longest_retweet_distribution_result_arrange = list()
count = 0
for every_graph in all_predict_json_file_dir:#29
    tag_result_template = {}
    retweet_count_high_result_template = {}
    retweet_count_middle_result_template = {}
    retweet_count_low_result_template = {}
    shortest_retweet_distribution_result_template = {}
    shorter_retweet_distribution_result_template = {}
    middle_retweet_distribution_result_template = {}
    longer_retweet_distribution_result_template = {}
    longest_retweet_distribution_result_template = {}
    graph_number = re.split('/',every_graph)[-2]
    #print('graph number: ',graph_number)
    for every_time_batch in range(10):#10
        for it in os.scandir(every_graph+str(every_time_batch)):    #需要蒐集幾個test結果
            if it.is_file():
                file_name = re.split('/|\\\\',it.path)[-1]
                file_name_part = re.split('/|\\\\|_',it.path)[10:]
                if (file_name_part[-1][-12:] == 'result.json' ):
                    if(file_name_part[0] == 'tag'):
                        tag_result = open(every_graph+str(every_time_batch)+'/'+file_name)           #把tag_result給讀入
                        ever_graph_result = json.load(tag_result)
                        tag_result.close()
                        tag_result_template[str(graph_number)+'_'+str(every_time_batch)+'_round'] = ever_graph_result
                    elif(file_name_part[0] == 'high'):
                        retweet_count_result = open(every_graph+str(every_time_batch)+'/'+file_name)           #把retweet_count_result給讀入
                        ever_graph_result = json.load(retweet_count_result)
                        retweet_count_result.close()
                        retweet_count_high_result_template[str(graph_number)+'_'+str(every_time_batch)] = ever_graph_result
                    elif(file_name_part[0] == 'middle'):
                        retweet_count_result = open(every_graph+str(every_time_batch)+'/'+file_name)           #把retweet_count_result給讀入
                        ever_graph_result = json.load(retweet_count_result)
                        retweet_count_result.close()
                        retweet_count_middle_result_template[str(graph_number)+'_'+str(every_time_batch)] = ever_graph_result
                    elif(file_name_part[0] == 'low'):
                        retweet_count_result = open(every_graph+str(every_time_batch)+'/'+file_name)           #把retweet_count_result給讀入
                        ever_graph_result = json.load(retweet_count_result)
                        retweet_count_result.close()
                        retweet_count_low_result_template[str(graph_number)+'_'+str(every_time_batch)] = ever_graph_result
                    else:
                        retweet_distribution_result = open(every_graph+str(every_time_batch)+'/'+file_name)           #把retweet_distribution_result給讀入
                        ever_graph_result = json.load(retweet_distribution_result)
                        retweet_distribution_result.close()
                        if(file_name_part[0] == 'in'):
                            shortest_retweet_distribution_result_template[str(graph_number)+'_'+str(every_time_batch)] = ever_graph_result
                        elif(file_name_part[1] == '12'):
                            shorter_retweet_distribution_result_template[str(graph_number)+'_'+str(every_time_batch)] = ever_graph_result
                        elif(file_name_part[1] == '24'):
                            middle_retweet_distribution_result_template[str(graph_number)+'_'+str(every_time_batch)] = ever_graph_result
                        elif(file_name_part[1] == '48'):
                            longer_retweet_distribution_result_template[str(graph_number)+'_'+str(every_time_batch)] = ever_graph_result
                        else:
                            longest_retweet_distribution_result_template[str(graph_number)+'_'+str(every_time_batch)] = ever_graph_result
    all_tag_result_arrange.append(pd.DataFrame.from_dict(tag_result_template))
    all_high_retweet_count_result_arrange.append(pd.DataFrame.from_dict(retweet_count_high_result_template))
    all_middle_retweet_count_result_arrange.append(pd.DataFrame.from_dict(retweet_count_middle_result_template))
    all_low_retweet_count_result_arrange.append(pd.DataFrame.from_dict(retweet_count_low_result_template))
    if bool(shortest_retweet_distribution_result_template):
        all_shortest_retweet_distribution_result_arrange.append(pd.DataFrame.from_dict(shortest_retweet_distribution_result_template))
        all_shorter_retweet_distribution_result_arrange.append(pd.DataFrame.from_dict(shorter_retweet_distribution_result_template))
        all_middle_retweet_distribution_result_arrange.append(pd.DataFrame.from_dict(middle_retweet_distribution_result_template))
        all_longer_retweet_distribution_result_arrange.append(pd.DataFrame.from_dict(longer_retweet_distribution_result_template))
        all_longest_retweet_distribution_result_arrange.append(pd.DataFrame.from_dict(longest_retweet_distribution_result_template))
    # if (graph_number == '15'):
    #     break
    # print('tag result: \n',all_tag_result_arrange)
    # print('retweet count result: \n',all_retweet_count_result_arrange)
    # print('retweet distribution result: \n',all_retweet_distribution_result_arrange)

with pd.ExcelWriter(elon_mask_test_graph_data_dir+'tags_result_rewrite.xlsx') as writer:
    for tag_result_number in range(len(all_tag_result_arrange)):
        graph_num = re.split('_',all_tag_result_arrange[tag_result_number].columns.values[0])
        all_tag_result_arrange[tag_result_number].to_excel(writer, sheet_name='tag_for_'+str(graph_num[0]))

with pd.ExcelWriter(elon_mask_test_graph_data_dir+'high_retweet_count_result_rewrite.xlsx') as writer:
    for retweet_count_result_number in range(len(all_high_retweet_count_result_arrange)):
        graph_num = re.split('_',all_high_retweet_count_result_arrange[retweet_count_result_number].columns.values[0])[0]
        all_high_retweet_count_result_arrange[retweet_count_result_number].to_excel(writer, sheet_name='count_for_'+str(graph_num))
with pd.ExcelWriter(elon_mask_test_graph_data_dir+'middle_retweet_count_result_rewrite.xlsx') as writer:
    for retweet_count_result_number in range(len(all_middle_retweet_count_result_arrange)):
        graph_num = re.split('_',all_middle_retweet_count_result_arrange[retweet_count_result_number].columns.values[0])[0]
        all_middle_retweet_count_result_arrange[retweet_count_result_number].to_excel(writer, sheet_name='count_for_'+str(graph_num))
with pd.ExcelWriter(elon_mask_test_graph_data_dir+'low_retweet_count_result_rewrite.xlsx') as writer:
    for retweet_count_result_number in range(len(all_low_retweet_count_result_arrange)):
        graph_num = re.split('_',all_low_retweet_count_result_arrange[retweet_count_result_number].columns.values[0])[0]
        all_low_retweet_count_result_arrange[retweet_count_result_number].to_excel(writer, sheet_name='count_for_'+str(graph_num))

with pd.ExcelWriter(elon_mask_test_graph_data_dir+'shortest_retweet_distribution_result_rewrite.xlsx') as writer:
    for retweet_distribution_number in range(len(all_shortest_retweet_distribution_result_arrange)):
        graph_num = re.split('_',all_shortest_retweet_distribution_result_arrange[retweet_distribution_number].columns.values[0])[0]
        all_shortest_retweet_distribution_result_arrange[retweet_distribution_number].to_excel(writer, sheet_name='distribution_for_'+str(graph_num))
with pd.ExcelWriter(elon_mask_test_graph_data_dir+'shorter_retweet_distribution_result_rewrite.xlsx') as writer:
    for retweet_distribution_number in range(len(all_shorter_retweet_distribution_result_arrange)):
        graph_num = re.split('_',all_shorter_retweet_distribution_result_arrange[retweet_distribution_number].columns.values[0])[0]
        all_shorter_retweet_distribution_result_arrange[retweet_distribution_number].to_excel(writer, sheet_name='distribution_for_'+str(graph_num))
with pd.ExcelWriter(elon_mask_test_graph_data_dir+'middle_retweet_distribution_result_rewrite.xlsx') as writer:
    for retweet_distribution_number in range(len(all_middle_retweet_distribution_result_arrange)):
        graph_num = re.split('_',all_middle_retweet_distribution_result_arrange[retweet_distribution_number].columns.values[0])[0]
        all_middle_retweet_distribution_result_arrange[retweet_distribution_number].to_excel(writer, sheet_name='distribution_for_'+str(graph_num))
with pd.ExcelWriter(elon_mask_test_graph_data_dir+'longer_retweet_distribution_result_rewrite.xlsx') as writer:
    for retweet_distribution_number in range(len(all_longer_retweet_distribution_result_arrange)):
        graph_num = re.split('_',all_longer_retweet_distribution_result_arrange[retweet_distribution_number].columns.values[0])[0]
        all_longer_retweet_distribution_result_arrange[retweet_distribution_number].to_excel(writer, sheet_name='distribution_for_'+str(graph_num))
with pd.ExcelWriter(elon_mask_test_graph_data_dir+'longest_retweet_distribution_result_rewrite.xlsx') as writer:
    for retweet_distribution_number in range(len(all_longest_retweet_distribution_result_arrange)):
        graph_num = re.split('_',all_longest_retweet_distribution_result_arrange[retweet_distribution_number].columns.values[0])[0]
        all_longest_retweet_distribution_result_arrange[retweet_distribution_number].to_excel(writer, sheet_name='distribution_for_'+str(graph_num))

    
#-----------------------------------------------------------------!!!!暫時用不到!!!!----------------------------------------------------------------------------------------------------------
# all_tweet_label_file = []                                                                                 #各個basegraph的label加入當feature
# for path in os.listdir(user_activelabel_dir):
#     if os.path.isfile(os.path.join(user_activelabel_dir, path)):
#         all_tweet_label_file.append(path)

# for file in all_tweet_label_file:
#     number = file.split("_")
#     number = number[2].split(".")
#     if not os.path.isdir(base_graph_dir+number[0]):
#         os.mkdir(base_graph_dir+number[0])
#     active_sitution = np.load(user_activelabel_dir+file)
#     user_feature = np.load(elon_mask_base_graph_data_dir+'user_feature.npy')
#     new_user_feature = np.column_stack([user_feature,active_sitution])
#     np.save(base_graph_dir+number[0]+"/base_graph_feature",new_user_feature)

#---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
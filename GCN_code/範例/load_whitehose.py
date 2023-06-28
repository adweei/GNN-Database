import pandas as pd
import os 
import dgl
import dgl.function as fn
from dgl.data.utils import save_graphs
import torch as th
import random

def remove_element(origin,delete):
    for element in delete:
        origin.remove(element)
    return origin

file_path = "D:/GNN/code/學長的code/Tweet31_30hr-20230228T051847Z-001/Tweet31_30hr/"
file_name = "WhiteHouse.cites"
weight_file_name = "weight.txt"
content_file_name = "WhiteHouse0.content"
label_path = "D:/GNN/code/label/whitehose_Tweet31_30"
edge = pd.read_csv(file_path+file_name,sep="\t",header=None)
edge_weight = pd.read_csv(file_path+weight_file_name,sep="\t",header=None)
node_feature = pd.read_csv(file_path+content_file_name,sep="\t",header=None)
#找出最大最小值 edge.max().max()    edge.min().min()

values_count1 = edge[0].value_counts()
values_count2 = edge[1].value_counts()
all_values_count = pd.Series(list(values_count1.index) + list(values_count2.index))#把edge中不重複的數字數量找出來
all_values_count = all_values_count.value_counts().sort_index()
all_number_convert = list(range(0,len(all_values_count)))#生成轉換用數列
all_number_convert_list = dict()

for i in range(0,len(all_values_count),1):#將對應數列表生成dict以供查詢
    all_number_convert_list[all_values_count.index[i]] = all_number_convert[i]

for row in range(len(edge[0])):#將原本跳號的vertx重新編號後存回edge變數中
    edge[0][row] = all_number_convert_list[edge[0][row]]
    edge[1][row] = all_number_convert_list[edge[1][row]]

edge.to_csv(file_path+"restruct_WhiteHouse.cites",sep="\t",header=None, index=False)#輸出備用

u, v = th.tensor(list(edge[0])), th.tensor(list(edge[1]))
base_graph = dgl.graph((u,v))#根據edge list建圖
edge_weight_list = []
for i in edge_weight[0]: 
    edge_weight_list.append([i])
base_graph.edata['weight'] = th.tensor(edge_weight_list)#對應的weight的放入
base_graph_feature = th.ones(base_graph.num_nodes(), len(node_feature.columns)-2)#node feature的初始化
base_graph_label = th.zeros(base_graph.num_nodes())

for node_number_feature in range(len(node_feature[0])):
    base_graph_feature[all_number_convert_list[int(node_feature.iloc[node_number_feature][0])]] = th.tensor(node_feature.iloc[node_number_feature][1:-1].tolist())#對應的feature經過查表 把feature寫入正確的位置
    base_graph_label[all_number_convert_list[int(node_feature.iloc[node_number_feature][0])]] = th.tensor(int(node_feature.iloc[node_number_feature][-1:]))#相應位置的label放入

base_graph.ndata['feature'] = base_graph_feature
base_graph.ndata['label'] = base_graph_label.type(th.LongTensor)

test_number = int(base_graph.num_nodes()*0.2)#前9次為278 最後一次為280
train_number = base_graph.num_nodes() - test_number#將dataset分為 8:2 然後每份資料都要成為test  所以跑9+1次
used_test_number = list()
leaf_orgin_number = all_number_convert.copy()
for round in range(0,3,1):#先挑選本次用到的數字 將剩下的在分為訓練和驗證
    copy_all_number_convert = all_number_convert.copy()
    now_used_number = random.sample(leaf_orgin_number,test_number)
    now_not_used_number = remove_element(copy_all_number_convert,now_used_number)
    leaf_orgin_number = remove_element(leaf_orgin_number,now_used_number)
    used_test_number.append(now_used_number)
    train_mask = [False for i in range(base_graph.num_nodes())]
    test_mask = [False for i in range(base_graph.num_nodes())]
    for nodes in range(0,base_graph.num_nodes(),1):
        if (nodes in now_not_used_number):
            train_mask[nodes] = True
            now_not_used_number.remove(nodes)
        else:
            test_mask[nodes] = True
            now_used_number.remove(nodes)
    df_train_mask = pd.DataFrame(train_mask)#將結果存檔到label資料夾
    df_test_mask = pd.DataFrame(test_mask) 
    df_train_mask.to_csv(label_path+'/train/train_label_'+str(round)+'.txt',sep='\n',index=False,header=False)
    df_test_mask.to_csv(label_path+'/test/test_label_'+str(round)+'.txt',sep='\n',index=False,header=False)
last_round_number = remove_element(all_number_convert,leaf_orgin_number)#因為test的數量有變化  所以額外處理一次
train_mask = [False for i in range(base_graph.num_nodes())]
test_mask = [False for i in range(base_graph.num_nodes())]
for nodes in range(0,base_graph.num_nodes(),1):
    if (nodes in last_round_number):
        train_mask[nodes] = True
        last_round_number.remove(nodes)
    else:
        test_mask[nodes] = True
        leaf_orgin_number.remove(nodes)
df_train_mask = pd.DataFrame(train_mask)
df_test_mask = pd.DataFrame(test_mask) 
df_train_mask.to_csv(label_path+'/train/train_label_'+str(round+1)+'.txt',sep='\n',index=False,header=False)
df_test_mask.to_csv(label_path+'/test/test_label_'+str(round+1)+'.txt',sep='\n',index=False,header=False)

save_graphs("D:/GNN/code/graph/whitehose_Tweet31_30.bin", [base_graph])#將graph的結果輸出
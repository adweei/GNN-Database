import os
import json
import pandas as pd

elon_mask_base_graph_data_dir = 'D:/GCN_Twitter/ElonMusk/2023-02-16/'  

#GAT(No edge weight)整理訓練結果，紀錄accuracy、precision、recall、F1

all_model_json_file_dir = list()
GAT_all_result_arrange = list()
#需要蒐集幾個model結果
for it in os.scandir("D:/GCN_Twitter/ElonMusk/2023-02-16/base_graph_for_model/GAT_FocalLoss1013/"):    
    if it.is_dir():
        all_model_json_file_dir.append(it.path + "/")
# print(all_model_json_file_dir)
for every_graph in range(1):
    GAT_result_arrange = {}
    for every_batch in range(5):#5
    #把GAT的user id給讀入
        GAT_result = open(all_model_json_file_dir[every_graph] + str(every_batch) + '_round_result_GAT_FocalLoss.json')            
        GAT_ever_graph_result = json.load(GAT_result)
        GAT_result.close()
        GAT_result_arrange[every_batch] = GAT_ever_graph_result
    GAT_all_result_arrange.append(pd.DataFrame.from_dict(GAT_result_arrange))
          
with pd.ExcelWriter(elon_mask_base_graph_data_dir + '_model_result_modify_collection_GAT_FocalLoss1014.xlsx') as writer:
    for model_result in range(len(GAT_all_result_arrange)):
        # print(GAT_all_result_arrange[model_result], model_result)
        GAT_all_result_arrange[model_result].to_excel(writer, sheet_name = 'model_for_' + str(model_result))
import os
import json
import json
import pandas as pd
import os
import re
from dgl.data.utils import load_graphs
from dgl.data.utils import save_graphs
# import torch.nn.functional as F 


elon_mask_base_graph_data_dir = 'D:/GCN_Twitter/ElonMusk/2023-02-16/'          #D:/GNN/new_data/MyResearch-main/MyResearch-main/ElonMusk/2023-05-01/
elon_mask_test_graph_data_dir = 'D:/GCN_Twitter/ElonMusk/2023-02-20/'          #D:/GNN/new_data/MyResearch-main/MyResearch-main/ElonMusk/2023-05-06/
predict_graph_dir = elon_mask_test_graph_data_dir + 'data_graph/'
#2/16~20的預測結果蒐集
all_predict_json_file_dir = list()
for it in os.scandir(predict_graph_dir):    #需要蒐集幾個model結果
    if it.is_dir():
        all_predict_json_file_dir.append(it.path + "/")
# print(all_predict_json_file_dir)

# 相同特徵
all_tag_result_arrange = list()
# 轉推數高中低
all_high_retweet_count_result_arrange = list()
all_high2_retweet_count_result_arrange = list()
all_middle_retweet_count_result_arrange = list()
all_low_retweet_count_result_arrange = list()
# 不固定特徵的轉推時間分布
all_first_retweet_distribution_result_arrange = list()
all_second_retweet_distribution_result_arrange = list()
all_third_retweet_distribution_result_arrange = list()
all_forth_retweet_distribution_result_arrange = list()
all_fifth_retweet_distribution_result_arrange = list()
all_sixth_retweet_distribution_result_arrange = list()
# 固定特徵的轉推時間分布
all_first_fixed_feature_retweet_distribution_result_arrange = list()
all_second_fixed_feature_retweet_distribution_result_arrange = list()
all_third_fixed_feature_retweet_distribution_result_arrange = list()
all_forth_fixed_feature_retweet_distribution_result_arrange = list()
all_fifth_fixed_feature_retweet_distribution_result_arrange = list()
all_sixth_fixed_feature_retweet_distribution_result_arrange = list()

for every_graph in all_predict_json_file_dir:#29
    # 相同特徵
    tag_result_template = {}
    # 轉推數高中低
    retweet_count_high_result_template = {}
    retweet_count_high2_result_template = {}
    retweet_count_middle_result_template = {}
    retweet_count_low_result_template = {}
    # 不固定特徵的轉推時間分布
    first_retweet_distribution_result_template = {}
    second_retweet_distribution_result_template = {}
    third_retweet_distribution_result_template = {}
    forth_retweet_distribution_result_template = {}    
    fifth_retweet_distribution_result_template = {}
    sixth_retweet_distribution_result_template = {}
    # 固定特徵的轉推時間分布
    first_fixed_feature_retweet_distribution_result_template = {}
    second_fixed_feature_retweet_distribution_result_template = {}
    third_fixed_feature_retweet_distribution_result_template = {}
    forth_fixed_feature_retweet_distribution_result_template = {}    
    fifth_fixed_feature_retweet_distribution_result_template = {}
    sixth_fixed_feature_retweet_distribution_result_template = {}

    graph_number = re.split('/', every_graph)[-2]
    # print('graph number: ', graph_number)
    for every_time_batch in range(10):#10
        for it in os.scandir(every_graph + str(every_time_batch)):    #需要蒐集幾個test結果
            if it.is_file():
                file_name = re.split('/|\\\\', it.path)[-1]
                # print(file_name)
                file_name_part = re.split('/|\\\\|_', it.path)[10:]
                # 為啥要[-12:]
                if (file_name_part[-1][-12:] == 'result.json' ):
                    # print(file_name_part[-3])
                    # 篩選固定特徵的結果
                    if (file_name_part[-3] == 'feature'):
                        retweet_distribution_result = open(every_graph + str(every_time_batch) + '/' + file_name)           #把retweet_distribution_result給讀入
                        ever_graph_result = json.load(retweet_distribution_result)
                        retweet_distribution_result.close()
                        #找in為開頭的檔名
                        if(file_name_part[0] == 'in'):
                            first_fixed_feature_retweet_distribution_result_template[str(graph_number)+'_'+str(every_time_batch)] = ever_graph_result
                        #找between', '25'為開頭的檔名    
                        elif(file_name_part[1] == '25'):
                            second_fixed_feature_retweet_distribution_result_template[str(graph_number)+'_'+str(every_time_batch)] = ever_graph_result
                        #找between', '31'為開頭的檔名      
                        elif(file_name_part[1] == '31'):
                            third_fixed_feature_retweet_distribution_result_template[str(graph_number)+'_'+str(every_time_batch)] = ever_graph_result
                        #找between', '50'為開頭的檔名      
                        elif(file_name_part[1] == '50'):
                            forth_fixed_feature_retweet_distribution_result_template[str(graph_number)+'_'+str(every_time_batch)] = ever_graph_result
                        #找between', '60'為開頭的檔名     
                        elif(file_name_part[1] == '60'):
                            fifth_fixed_feature_retweet_distribution_result_template[str(graph_number)+'_'+str(every_time_batch)] = ever_graph_result 
                        else:
                            sixth_fixed_feature_retweet_distribution_result_template[str(graph_number)+'_'+str(every_time_batch)] = ever_graph_result
                    else:
                        if(file_name_part[0] == 'tag'):
                            #把tag_result給讀入
                            tag_result = open(every_graph + str(every_time_batch) + '/' + file_name)     
                            ever_graph_result = json.load(tag_result)
                            tag_result.close()
                            # print(ever_graph_result)
                            tag_result_template[str(graph_number) + '_' + str(every_time_batch) + '_round'] = ever_graph_result

                        elif(file_name_part[0] == 'high'):
                            retweet_count_result = open(every_graph + str(every_time_batch) + '/' + file_name)           #把retweet_count_result給讀入
                            ever_graph_result = json.load(retweet_count_result)
                            retweet_count_result.close()
                            retweet_count_high_result_template[str(graph_number) + '_' + str(every_time_batch)] = ever_graph_result
                        elif(file_name_part[0] == 'high2'):
                            retweet_count_result = open(every_graph + str(every_time_batch) + '/' + file_name)           #把retweet_count_result給讀入
                            ever_graph_result = json.load(retweet_count_result)
                            retweet_count_result.close()
                            retweet_count_high2_result_template[str(graph_number) + '_' + str(every_time_batch)] = ever_graph_result
                        elif(file_name_part[0] == 'middle'):
                            retweet_count_result = open(every_graph + str(every_time_batch) + '/' + file_name)           #把retweet_count_result給讀入
                            ever_graph_result = json.load(retweet_count_result)
                            retweet_count_result.close()
                            retweet_count_middle_result_template[str(graph_number) + '_' + str(every_time_batch)] = ever_graph_result
                        elif(file_name_part[0] == 'low'):
                            retweet_count_result = open(every_graph + str(every_time_batch) + '/' + file_name)           #把retweet_count_result給讀入
                            ever_graph_result = json.load(retweet_count_result)
                            retweet_count_result.close()
                            retweet_count_low_result_template[str(graph_number) + '_' + str(every_time_batch)] = ever_graph_result
                        else:
                            retweet_distribution_result = open(every_graph + str(every_time_batch) + '/' + file_name)           #把retweet_distribution_result給讀入
                            ever_graph_result = json.load(retweet_distribution_result)
                            retweet_distribution_result.close()
                            #找in為開頭的檔名
                            if(file_name_part[0] == 'in'):
                                first_retweet_distribution_result_template[str(graph_number)+'_'+str(every_time_batch)] = ever_graph_result
                            #找between', '25'為開頭的檔名    
                            elif(file_name_part[1] == '25'):
                                second_retweet_distribution_result_template[str(graph_number)+'_'+str(every_time_batch)] = ever_graph_result
                            #找between', '31'為開頭的檔名      
                            elif(file_name_part[1] == '31'):
                                third_retweet_distribution_result_template[str(graph_number)+'_'+str(every_time_batch)] = ever_graph_result
                            #找between', '50'為開頭的檔名      
                            elif(file_name_part[1] == '50'):
                                forth_retweet_distribution_result_template[str(graph_number)+'_'+str(every_time_batch)] = ever_graph_result
                            #找between', '60'為開頭的檔名     
                            elif(file_name_part[1] == '60'):
                                fifth_retweet_distribution_result_template[str(graph_number)+'_'+str(every_time_batch)] = ever_graph_result 
                            else:
                                sixth_retweet_distribution_result_template[str(graph_number)+'_'+str(every_time_batch)] = ever_graph_result
    all_tag_result_arrange.append(pd.DataFrame.from_dict(tag_result_template))
    all_high_retweet_count_result_arrange.append(pd.DataFrame.from_dict(retweet_count_high_result_template))
    all_high2_retweet_count_result_arrange.append(pd.DataFrame.from_dict(retweet_count_high2_result_template))
    all_middle_retweet_count_result_arrange.append(pd.DataFrame.from_dict(retweet_count_middle_result_template))
    all_low_retweet_count_result_arrange.append(pd.DataFrame.from_dict(retweet_count_low_result_template))
    if bool(first_retweet_distribution_result_template):
        all_first_retweet_distribution_result_arrange.append(pd.DataFrame.from_dict(first_retweet_distribution_result_template))
        all_second_retweet_distribution_result_arrange.append(pd.DataFrame.from_dict(second_retweet_distribution_result_template))
        all_third_retweet_distribution_result_arrange.append(pd.DataFrame.from_dict(third_retweet_distribution_result_template))
        all_forth_retweet_distribution_result_arrange.append(pd.DataFrame.from_dict(forth_retweet_distribution_result_template))
        all_fifth_retweet_distribution_result_arrange.append(pd.DataFrame.from_dict(fifth_retweet_distribution_result_template))
        all_sixth_retweet_distribution_result_arrange.append(pd.DataFrame.from_dict(sixth_retweet_distribution_result_template))
    if bool(first_fixed_feature_retweet_distribution_result_template):
        all_first_fixed_feature_retweet_distribution_result_arrange.append(pd.DataFrame.from_dict(first_fixed_feature_retweet_distribution_result_template))
        all_second_fixed_feature_retweet_distribution_result_arrange.append(pd.DataFrame.from_dict(second_fixed_feature_retweet_distribution_result_template))
        all_third_fixed_feature_retweet_distribution_result_arrange.append(pd.DataFrame.from_dict(third_fixed_feature_retweet_distribution_result_template))
        all_forth_fixed_feature_retweet_distribution_result_arrange.append(pd.DataFrame.from_dict(forth_fixed_feature_retweet_distribution_result_template))
        all_fifth_fixed_feature_retweet_distribution_result_arrange.append(pd.DataFrame.from_dict(fifth_fixed_feature_retweet_distribution_result_template))
        all_sixth_fixed_feature_retweet_distribution_result_arrange.append(pd.DataFrame.from_dict(sixth_fixed_feature_retweet_distribution_result_template)) 


with pd.ExcelWriter(elon_mask_test_graph_data_dir + 'prediction_result/tags_result_rewrite.xlsx') as writer:
    for tag_result_number in range(len(all_tag_result_arrange)):
        tag_result_num = all_tag_result_arrange[tag_result_number]
        graph_num = re.split('_', tag_result_num.columns.values[0])
        all_tag_result_arrange[tag_result_number].to_excel(writer, sheet_name = 'tag_for_' + str(graph_num[0]))

with pd.ExcelWriter(elon_mask_test_graph_data_dir + 'prediction_result/high_retweet_count_result_rewrite.xlsx') as writer:
    for retweet_count_result_number in range(len(all_high_retweet_count_result_arrange)):
        graph_num = re.split('_',all_high_retweet_count_result_arrange[retweet_count_result_number].columns.values[0])[0]
        all_high_retweet_count_result_arrange[retweet_count_result_number].to_excel(writer, sheet_name = 'count_for_' + str(graph_num))

with pd.ExcelWriter(elon_mask_test_graph_data_dir + 'prediction_result/high2_retweet_count_result_rewrite.xlsx') as writer:
    for retweet_count_result_number in range(len(all_high2_retweet_count_result_arrange)):
        graph_num = re.split('_',all_high2_retweet_count_result_arrange[retweet_count_result_number].columns.values[0])[0]
        all_high2_retweet_count_result_arrange[retweet_count_result_number].to_excel(writer, sheet_name = 'count_for_' + str(graph_num))

with pd.ExcelWriter(elon_mask_test_graph_data_dir + 'prediction_result/middle_retweet_count_result_rewrite.xlsx') as writer:
    for retweet_count_result_number in range(len(all_middle_retweet_count_result_arrange)):
        # print(all_middle_retweet_count_result_arrange[retweet_count_result_number])
        graph_num = re.split('_',all_middle_retweet_count_result_arrange[retweet_count_result_number].columns.values[0])[0]
        all_middle_retweet_count_result_arrange[retweet_count_result_number].to_excel(writer, sheet_name='count_for_'+str(graph_num))

with pd.ExcelWriter(elon_mask_test_graph_data_dir + 'prediction_result/low_retweet_count_result_rewrite.xlsx') as writer:
    for retweet_count_result_number in range(len(all_low_retweet_count_result_arrange)):
        graph_num = re.split('_',all_low_retweet_count_result_arrange[retweet_count_result_number].columns.values[0])[0]
        all_low_retweet_count_result_arrange[retweet_count_result_number].to_excel(writer, sheet_name='count_for_'+str(graph_num))

with pd.ExcelWriter(elon_mask_test_graph_data_dir + 'prediction_result/first_retweet_distribution_result_rewrite.xlsx') as writer:
    for retweet_distribution_number in range(len(all_first_retweet_distribution_result_arrange)):
        graph_num = re.split('_',all_first_retweet_distribution_result_arrange[retweet_distribution_number].columns.values[0])[0]
        all_first_retweet_distribution_result_arrange[retweet_distribution_number].to_excel(writer, sheet_name='distribution_for_' + str(graph_num))

with pd.ExcelWriter(elon_mask_test_graph_data_dir + 'prediction_result/second_retweet_distribution_result_rewrite.xlsx') as writer:
    for retweet_distribution_number in range(len(all_second_retweet_distribution_result_arrange)):
        graph_num = re.split('_',all_second_retweet_distribution_result_arrange[retweet_distribution_number].columns.values[0])[0]
        all_second_retweet_distribution_result_arrange[retweet_distribution_number].to_excel(writer, sheet_name='distribution_for_'+str(graph_num))

with pd.ExcelWriter(elon_mask_test_graph_data_dir + 'prediction_result/third_retweet_distribution_result_rewrite.xlsx') as writer:
    for retweet_distribution_number in range(len(all_third_retweet_distribution_result_arrange)):
        graph_num = re.split('_',all_third_retweet_distribution_result_arrange[retweet_distribution_number].columns.values[0])[0]
        all_third_retweet_distribution_result_arrange[retweet_distribution_number].to_excel(writer, sheet_name='distribution_for_'+str(graph_num))

with pd.ExcelWriter(elon_mask_test_graph_data_dir + 'prediction_result/forth_retweet_distribution_result_rewrite.xlsx') as writer:
    for retweet_distribution_number in range(len(all_forth_retweet_distribution_result_arrange)):
        graph_num = re.split('_',all_forth_retweet_distribution_result_arrange[retweet_distribution_number].columns.values[0])[0]
        all_forth_retweet_distribution_result_arrange[retweet_distribution_number].to_excel(writer, sheet_name='distribution_for_'+str(graph_num))

with pd.ExcelWriter(elon_mask_test_graph_data_dir + 'prediction_result/fifth_retweet_distribution_result_rewrite.xlsx') as writer:
    for retweet_distribution_number in range(len(all_fifth_retweet_distribution_result_arrange)):
        graph_num = re.split('_',all_fifth_retweet_distribution_result_arrange[retweet_distribution_number].columns.values[0])[0]
        all_fifth_retweet_distribution_result_arrange[retweet_distribution_number].to_excel(writer, sheet_name='distribution_for_'+str(graph_num))

with pd.ExcelWriter(elon_mask_test_graph_data_dir + 'prediction_result/sixth_retweet_distribution_result_rewrite.xlsx') as writer:
    for retweet_distribution_number in range(len(all_sixth_retweet_distribution_result_arrange)):
        graph_num = re.split('_',all_sixth_retweet_distribution_result_arrange[retweet_distribution_number].columns.values[0])[0]
        all_sixth_retweet_distribution_result_arrange[retweet_distribution_number].to_excel(writer, sheet_name='distribution_for_'+str(graph_num))       

with pd.ExcelWriter(elon_mask_test_graph_data_dir + 'prediction_result/first_fixed_feature_retweet_distribution_result_rewrite.xlsx') as writer:
    for retweet_distribution_number in range(len(all_first_fixed_feature_retweet_distribution_result_arrange)):
        graph_num = re.split('_',all_first_fixed_feature_retweet_distribution_result_arrange[retweet_distribution_number].columns.values[0])[0]
        all_first_fixed_feature_retweet_distribution_result_arrange[retweet_distribution_number].to_excel(writer, sheet_name='distribution_for_' + str(graph_num))

with pd.ExcelWriter(elon_mask_test_graph_data_dir + 'prediction_result/second_fixed_feature_retweet_distribution_result_rewrite.xlsx') as writer:
    for retweet_distribution_number in range(len(all_second_fixed_feature_retweet_distribution_result_arrange)):
        graph_num = re.split('_',all_second_fixed_feature_retweet_distribution_result_arrange[retweet_distribution_number].columns.values[0])[0]
        all_second_fixed_feature_retweet_distribution_result_arrange[retweet_distribution_number].to_excel(writer, sheet_name='distribution_for_'+str(graph_num))

with pd.ExcelWriter(elon_mask_test_graph_data_dir + 'prediction_result/third_fixed_feature_retweet_distribution_result_rewrite.xlsx') as writer:
    for retweet_distribution_number in range(len(all_third_fixed_feature_retweet_distribution_result_arrange)):
        graph_num = re.split('_',all_third_fixed_feature_retweet_distribution_result_arrange[retweet_distribution_number].columns.values[0])[0]
        all_third_fixed_feature_retweet_distribution_result_arrange[retweet_distribution_number].to_excel(writer, sheet_name='distribution_for_'+str(graph_num))

with pd.ExcelWriter(elon_mask_test_graph_data_dir + 'prediction_result/forth_fixed_feature_retweet_distribution_result_rewrite.xlsx') as writer:
    for retweet_distribution_number in range(len(all_forth_fixed_feature_retweet_distribution_result_arrange)):
        graph_num = re.split('_',all_forth_fixed_feature_retweet_distribution_result_arrange[retweet_distribution_number].columns.values[0])[0]
        all_forth_fixed_feature_retweet_distribution_result_arrange[retweet_distribution_number].to_excel(writer, sheet_name='distribution_for_'+str(graph_num))

with pd.ExcelWriter(elon_mask_test_graph_data_dir + 'prediction_result/fifth_fixed_feature_retweet_distribution_result_rewrite.xlsx') as writer:
    for retweet_distribution_number in range(len(all_fifth_fixed_feature_retweet_distribution_result_arrange)):
        graph_num = re.split('_',all_fifth_fixed_feature_retweet_distribution_result_arrange[retweet_distribution_number].columns.values[0])[0]
        all_fifth_fixed_feature_retweet_distribution_result_arrange[retweet_distribution_number].to_excel(writer, sheet_name='distribution_for_'+str(graph_num))

with pd.ExcelWriter(elon_mask_test_graph_data_dir + 'prediction_result/sixth_fixed_feature_retweet_distribution_result_rewrite.xlsx') as writer:
    for retweet_distribution_number in range(len(all_sixth_fixed_feature_retweet_distribution_result_arrange)):
        graph_num = re.split('_',all_sixth_fixed_feature_retweet_distribution_result_arrange[retweet_distribution_number].columns.values[0])[0]
        all_sixth_fixed_feature_retweet_distribution_result_arrange[retweet_distribution_number].to_excel(writer, sheet_name='distribution_for_'+str(graph_num))   
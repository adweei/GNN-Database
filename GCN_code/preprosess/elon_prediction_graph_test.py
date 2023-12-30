import sys
sys.path.append("D:/GCN_Twitter/GCN_code/gcn/")
from model_test import model_test
from gcn_model import Net
from dgl.data.utils import load_graphs
import torch as th
import re
import numpy as np
import time

start = time.time()
elon_mask_base_graph_data_dir = 'D:/GCN_Twitter/ElonMusk/2023-02-16/'        
elon_mask_test_graph_data_dir = 'D:/GCN_Twitter/ElonMusk/2023-02-20/'

#retweet count高中低 
def retweet_count(g, features, labels, retweet_distribution_mask, retweet_count_model_list):
    # print('number: ', number)
    for seq in range(4):
        net = Net(len(features[0]))
        print('model for retweet count is: ', retweet_count_model_list[seq][1])
        print('round of model for retweet count is: ', retweet_count_model_list[seq][2])
        net.load_state_dict(th.load(model_choose_file_dir + str(retweet_count_model_list[seq][1]) +
                                    '/' + str(retweet_count_model_list[seq][2]) + '_round_result_edge_weight_Adjustment.pth'))
        retweet_count_model = model_test()  
        print('model of  retweet count acc is: {:.6f}'.format(retweet_count_model.test_model(net, g, features, labels, retweet_distribution_mask))) 
        if seq == 0:
            retweet_count_model.model_save(data_graph_dir + str(number[0]) + "/" + str(time_point) + "/" + str(number[0]) +
                                          '_high_retweet_count_model_result')
            print('----------------------------------------------------------high retweet count---------------------------------------------------------------------')  
        elif seq == 1:
            retweet_count_model.model_save(data_graph_dir + str(number[0]) + "/" + str(time_point) + "/" + str(number[0]) +
                                          '_middle_retweet_count_model_result')
            print('-------------------------------------------------------middle retweet count-----------------------------------------------------------------')            
        elif seq == 2:
            retweet_count_model.model_save(data_graph_dir + str(number[0]) + "/" + str(time_point) + "/" + str(number[0]) +
                                          '_middle2_retweet_count_model_result')
            print('----------------------------------------------------------middle2 retweet count-------------------------------------------------------------------')
        elif seq == 3:
            retweet_count_model.model_save(data_graph_dir + str(number[0]) + "/" + str(time_point) + "/" + str(number[0]) +
                                          '_low_retweet_count_model_result')                                             
            print('----------------------------------------------------------low retweet count----------------------------------------------------------------------')
        del net

def retweet_time(g, features, labels, retweet_distribution_mask, distribution_model_list):
    for time in range(6):
        net = Net(len(features[0]))
        # print('model for longer time collected is: ',distribution_model_list[time][1])
        # print('round of model for longer time collected is: ',distribution_model_list[time][2])        
        net.load_state_dict(th.load(model_choose_file_dir + str(distribution_model_list[time][1]) + '/'+ 
                                    str(distribution_model_list[time][2]) + '_round_result_edge_weight_Adjustment.pth'))
        retweet_time_model = model_test()
        print('model of time acc is: {:.6f}'.format(retweet_time_model.test_model(net, g, features, labels, retweet_distribution_mask)))
        if time == 0:
            print('model for longer time collected is: ',distribution_model_list[time][1])
            print('round of model for longer time collected is: ',distribution_model_list[time][2])  
            retweet_time_model.model_save(data_graph_dir + str(number[0]) + "/" + 
                                        str(time_point) + "/" + str(number[0]) + '_over_84_model_result')
            print('----------------------------------------------------------over 84H---------------------------------------------------------------------')
        elif time == 1:
            print('model for longer time collected is: ',distribution_model_list[time][1])
            print('round of model for longer time collected is: ',distribution_model_list[time][2])              
            retweet_time_model.model_save(data_graph_dir + str(number[0]) + "/" + 
                                      str(time_point) + "/" + str(number[0]) + '_between_60_83_model_result')
            print('---------------------------------------------------between 60H to 83H------------------------------------------------------------------')
        elif time == 2:
            print('model for longer time collected is: ',distribution_model_list[time][1])
            print('round of model for longer time collected is: ',distribution_model_list[time][2])              
            retweet_time_model.model_save(data_graph_dir + str(number[0]) + "/" + 
                                      str(time_point) + "/" + str(number[0]) + '_between_50_59_model_result')
            print('---------------------------------------------------between 50H to 59H------------------------------------------------------------------')
        elif time == 3:
            print('model for longer time collected is: ',distribution_model_list[time][1])
            print('round of model for longer time collected is: ',distribution_model_list[time][2])              
            retweet_time_model.model_save(data_graph_dir + str(number[0]) + "/" + 
                                      str(time_point) + "/" + str(number[0]) + '_between_31_49_model_result')
            print('---------------------------------------------------between 31H to 49H------------------------------------------------------------------')
        elif time == 4:
            print('model for longer time collected is: ',distribution_model_list[time][1])
            print('round of model for longer time collected is: ',distribution_model_list[time][2])              
            retweet_time_model.model_save(data_graph_dir + str(number[0]) + "/" + 
                                      str(time_point) + "/" + str(number[0]) + '_between_25_30_model_result')  
            print('---------------------------------------------------between 25H to 30H------------------------------------------------------------------')
        elif time == 5:
            print('model for longer time collected is: ',distribution_model_list[time][1])
            print('round of model for longer time collected is: ',distribution_model_list[time][2])              
            retweet_time_model.model_save(data_graph_dir + str(number[0]) + "/" + 
                                      str(time_point) + "/" + str(number[0]) + '_in_12_model_result')    
            print('---------------------------------------------------under 24H---------------------------------------------------------------------------')          
        del net        
      
#-------------------------------------------將data graph的tag資訊一起讀出----------------------------------------------------------------
data_graph_dir = elon_mask_test_graph_data_dir + 'data_graph/'
data_graph = open(data_graph_dir + 'test_graph_tag.txt')           #神秘tag咒語，請手建
data_graph_list = list()
for line in data_graph:
    content = re.split(": | |\\n", line)
    data_graph_list.append(content[:3])
data_graph_list = np.array(data_graph_list)
#-------------------------------------------將tag要選擇的model給讀取出來--------------------------------------------------------------------
model_choose_file_dir = elon_mask_base_graph_data_dir + 'base_graph_for_model/'
tag_model = open(model_choose_file_dir + 'model_for_feature(duration).txt')
tag_model_list = list()
for line in tag_model:
    content = re.split(": | |\\n",line)
    tag_model_list.append(content[:4])
tag_model_list = np.array(tag_model_list)
#------------------------------------------將retweet 數 高、中、低 三個model是誰讀取出來------------------------------------------------------
retweet_count_model = open(model_choose_file_dir + 'model_for_retweet.txt')
retweet_count_model_list = list()
for line in retweet_count_model:
    content = re.split(": | |\\n",line)
    retweet_count_model_list.append(content[:3])
retweet_count_model_list = np.array(retweet_count_model_list)
# #-----------------------------------------將不同時間點要選的model讀取出來----------------------------------------------------------------
distribution_model = open(model_choose_file_dir + 'model_for_distribution.txt')
distribution_model_list = list()
for line in distribution_model:
    content = re.split(": | |\\n",line)
    distribution_model_list.append(content[:3])
distribution_model_list = np.array(distribution_model_list)
# #-----------------------------------------將固定關鍵字不同時間點要選的model讀取出來----------------------------------------------------------------
fixed_distribution_model = open(model_choose_file_dir + 'fixed_feature_model_for_distribution.txt')
fixed_distribution_model_list = list()
for line in fixed_distribution_model:
    content = re.split(": | |\\n",line)
    fixed_distribution_model_list.append(content[:3])
fixed_distribution_model_list = np.array(fixed_distribution_model_list)

# print("data graph 有這些: \n", data_graph_list)
# print("關鍵字model 選這些: \n", tag_model_list)
# print("轉推數高中低model選這些: \n", retweet_count_model_list)
print("轉推時間分布model選這些: \n", distribution_model_list)
# print('固定特徵轉推時間分布model選這些: \n', fixed_distribution_model_list)
#----------------------------------------將所有要跑結果的data graph執行------------------------------------------------------------------------

# count = 0
for number in data_graph_list:
    #----------------------------------------將轉推時間分佈讀取出來----------------------------------------------------------------------------------------------------
    retweet_distribution = np.load(elon_mask_base_graph_data_dir + 'retweet_distribution/'+
                                   str(number[0]) + '_graph_retweet_distribution_rewrite.npy')
    # print('retweet distribution: ', retweet_distribution[:,1])
    # print('number: ', data_graph_list)
    for time_point in range(10):
        retweet_distribution_mask = np.ones(retweet_distribution.shape[0])
        retweet_distribution_mask = retweet_distribution_mask - retweet_distribution[:, time_point]
        retweet_distribution_mask = th.from_numpy(retweet_distribution_mask).type(th.bool)
        # print('*****************************************************************************************************************************************************************************************')
        print("graph number: ",number[0])
        # print("now graph's time point: ",time_point)
        # print("graph tagt's tag: ",number[1])
        # print('graph\'s retweet distribution: \n',list(retweet_distribution[:,time_point]))
        # print('graph\'s retweet distribution mask: \n',list(retweet_distribution_mask))

#----------------------------------------------不同test graph挑不同feature-----------------------------------------------------------------------
        #tag 的結果
        if int(number[1]) == 1:
            #將選到的大類別存起來
            temp_tag_model_list = tag_model_list[np.where(tag_model_list[:,0] == (number[1]))]
        #     # print('data graph dir: ', data_graph_dir + str(number[0]) + "/" + str(time_point) + "/" + str(number[0]) + "_rewrite.bin")             
        #     # print('choose_tag_model_list: ', temp_tag_model_list[np.where(temp_tag_model_list[:,1] == (number[2]))]) 
              
            dataset,label = load_graphs(data_graph_dir + str(number[0]) + "/" + 
                                        str(time_point) + "/" + str(number[0]) + "_rewrite.bin")
            Graph = dataset[0]
            cuda_g = Graph.to('cuda:0')
            features = cuda_g.ndata['feature']
            labels = cuda_g.ndata['label']
            
            # GCN
            # net = Net(len(features[0]))
            # # print(tag_model_list[np.where(tag_model_list[:,0] == (number[1]))])
            # print("model number is: ", str(temp_tag_model_list[np.where(temp_tag_model_list[:,1] == (number[2]))][0][2]))
            # print("the result round of model is: ",str(temp_tag_model_list[np.where(temp_tag_model_list[:,1] == (number[2]))][0][3]))
            # # print("module graph dir: ", model_choose_file_dir + str(temp_tag_model_list[np.where(temp_tag_model_list[:,1] == (number[2]))][0][2]) + '/' +
            # #                             str(temp_tag_model_list[np.where(temp_tag_model_list[:,1] == (number[2]))][0][3]) +
            # #                             '_round_result_edge_weight_Adjustment.pth')
            # net.load_state_dict(th.load(model_choose_file_dir + str(temp_tag_model_list[np.where(temp_tag_model_list[:,1] == (number[2]))][0][2]) + '/' +
            #                             str(temp_tag_model_list[np.where(temp_tag_model_list[:,1] == (number[2]))][0][3]) +
            #                             '_round_result_edge_weight_Adjustment.pth'))
            # tag_test = model_test()
            # print('tag\'s GCN(tag = 1) acc is: {:.6f}'.format(tag_test.test_model(net, cuda_g, features, labels, retweet_distribution_mask)))
            # tag_test.model_save(data_graph_dir + str(number[0]) + "/" +
            #                     str(time_point) + "/" + str(number[0]) + '_tag_model_result')
            # del net
            print('-------------------------------------------------------------post own tweet----------------------------------------------------------------')
            # print('function_num: ', number)
            # retweet_count(cuda_g, features, labels, retweet_distribution_mask, retweet_count_model_list)
            retweet_time(cuda_g, features, labels, retweet_distribution_mask, distribution_model_list)
            
        elif int(number[1]) == 2:
            #將選到的大類別存起來
            temp_tag_model_list = tag_model_list[np.where(tag_model_list[:,0] == (number[1]))]
            # print('data graph dir: ', data_graph_dir + str(number[0]) + "/" + str(time_point) + "/" + str(number[0]) + "_rewrite.bin")             
            # print('choose_tag_model_list: ', temp_tag_model_list[np.where(temp_tag_model_list[:,1] == (number[2]))]) 
              
            dataset,label = load_graphs(data_graph_dir + str(number[0]) + "/" + 
                                        str(time_point) + "/" + str(number[0]) + "_rewrite.bin")
            Graph = dataset[0]
            cuda_g = Graph.to('cuda:0')
            features = cuda_g.ndata['feature']
            labels = cuda_g.ndata['label']
            
            # GCN
            # net = Net(len(features[0]))
            # print("model number is: ", str(temp_tag_model_list[np.where(temp_tag_model_list[:,1] == (number[2]))][0][2]))
            # print("the result round of model is: ",str(temp_tag_model_list[np.where(temp_tag_model_list[:,1] == (number[2]))][0][3]))
            # # print("module graph dir: ", model_choose_file_dir + str(temp_tag_model_list[np.where(temp_tag_model_list[:,1] == (number[2]))][0][2]) + '/' +
            # #                             str(temp_tag_model_list[np.where(temp_tag_model_list[:,1] == (number[2]))][0][3]) +
            # #                             '_round_result_edge_weight_Adjustment.pth')
            # net.load_state_dict(th.load(model_choose_file_dir + str(temp_tag_model_list[np.where(temp_tag_model_list[:,1] == (number[2]))][0][2]) + '/' +
            #                             str(temp_tag_model_list[np.where(temp_tag_model_list[:,1] == (number[2]))][0][3]) +
            #                             '_round_result_edge_weight_Adjustment.pth'))
            # tag_test = model_test()
            # print('tag\'s GCN(tag = 2) acc is: {:.6f}'.format(tag_test.test_model(net, cuda_g, features, labels, retweet_distribution_mask)))
            # tag_test.model_save(data_graph_dir + str(number[0]) + "/" +
            #                     str(time_point) + "/" + str(number[0]) + '_tag_model_result')
            # del net
            print('-------------------------------------------------------------reply someone tweet--------------------------------------------------------------')  
            # retweet_count(cuda_g, features, labels, retweet_distribution_mask, retweet_count_model_list)
            retweet_time(cuda_g, features, labels, retweet_distribution_mask, distribution_model_list)

        elif int(number[1]) == 3:
            #將選到的大類別存起來
            temp_tag_model_list = tag_model_list[np.where(tag_model_list[:,0] == (number[1]))]
            # print('data graph dir: ', data_graph_dir + str(number[0]) + "/" + str(time_point) + "/" + str(number[0]) + "_rewrite.bin")             
            # print('choose_tag_model_list: ', temp_tag_model_list[np.where(temp_tag_model_list[:,1] == (number[2]))]) 
              
            dataset,label = load_graphs(data_graph_dir + str(number[0]) + "/" + 
                                        str(time_point) + "/" + str(number[0]) + "_rewrite.bin")
            Graph = dataset[0]
            cuda_g = Graph.to('cuda:0')
            features = cuda_g.ndata['feature']
            labels = cuda_g.ndata['label']
            
            # GCN
            # net = Net(len(features[0]))
            # print("model number is: ", str(temp_tag_model_list[np.where(temp_tag_model_list[:,1] == (number[2]))][0][2]))
            # print("the result round of model is: ",str(temp_tag_model_list[np.where(temp_tag_model_list[:,1] == (number[2]))][0][3]))
            # # print("module graph dir: ", model_choose_file_dir + str(temp_tag_model_list[np.where(temp_tag_model_list[:,1] == (number[2]))][0][2]) + '/' +
            # #                             str(temp_tag_model_list[np.where(temp_tag_model_list[:,1] == (number[2]))][0][3]) +
            # #                             '_round_result_edge_weight_Adjustment.pth')
            # net.load_state_dict(th.load(model_choose_file_dir + str(temp_tag_model_list[np.where(temp_tag_model_list[:,1] == (number[2]))][0][2]) + '/' +
            #                             str(temp_tag_model_list[np.where(temp_tag_model_list[:,1] == (number[2]))][0][3]) +
            #                             '_round_result_edge_weight_Adjustment.pth'))
            # tag_test = model_test()
            # print('tag\'s GCN(tag = 3) acc is: {:.6f}'.format(tag_test.test_model(net, cuda_g, features, labels, retweet_distribution_mask)))
            # tag_test.model_save(data_graph_dir + str(number[0]) + "/" +
            #                     str(time_point) + "/" + str(number[0]) + '_tag_model_result')
            # del net
            print('-------------------------------------------------------------reply own tweet---------------------------------------------------------')
            # retweet_count(cuda_g, features, labels, retweet_distribution_mask, retweet_count_model_list)      
            retweet_time(cuda_g, features, labels, retweet_distribution_mask, distribution_model_list)

#---------------------------------------------------------------------------------------------------------------------------------------------------------
       


# print(count)
end = time.time()

total_time = end - start
sec = int(total_time % 60)
min = int(total_time / 60)
print("執行時間：%d分 %d 秒" %(min, sec))
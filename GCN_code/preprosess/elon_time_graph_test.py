import sys
sys.path.append("D:/GCN_Twitter/GCN_code/gcn/")
from model_test import model_test
from dgl.data.utils import load_graphs
import torch as th
from gcn_model import Net
import re
import numpy as np
import time

start = time.time()
elon_mask_base_graph_data_dir = 'D:/GCN_Twitter/ElonMusk/2023-02-16/'        
elon_mask_test_graph_data_dir = 'D:/GCN_Twitter/ElonMusk/2023-02-20/'
#-------------------------------------------將data graph的tag資訊一起讀出----------------------------------------------------------------
data_graph_dir = elon_mask_test_graph_data_dir + 'data_graph/'
data_graph = open(data_graph_dir + 'graph_tag01.txt')           #神秘tag咒語，請手建
data_graph_list = list()
for line in data_graph:
    content = re.split(": | |\\n",line)
    data_graph_list.append(content[:2])
data_graph_list = np.array(data_graph_list)
# print(data_graph_list)
#-------------------------------------------將tag要選擇的model給讀取出來--------------------------------------------------------------------
model_choose_file_dir = elon_mask_base_graph_data_dir + 'base_graph_for_model/'
tag_model = open(model_choose_file_dir+'model_for_tag.txt')
tag_model_list = list()
for line in tag_model:
    content = re.split(": | |\\n",line)
    tag_model_list.append(content[:3])
tag_model_list = np.array(tag_model_list)
#------------------------------------------將retweet 數 高、中、低 三個model是誰讀取出來------------------------------------------------------
retweet_count_model = open(model_choose_file_dir + 'model_for_retweet.txt')
retweet_count_model_list = list()
for line in retweet_count_model:
    content = re.split(": | |\\n",line)
    retweet_count_model_list.append(content[:3])
retweet_count_model_list = np.array(retweet_count_model_list)
#-----------------------------------------將關鍵字不同時間點要選的model讀取出來----------------------------------------------------------------
distribution_model = open(model_choose_file_dir + 'model_for_distribution.txt')
distribution_model_list = list()
for line in distribution_model:
    content = re.split(": | |\\n",line)
    distribution_model_list.append(content[:3])
distribution_model_list = np.array(distribution_model_list)
# print("data graph 有這些: \n",data_graph_list)
# print("關鍵字model 選這些: \n",tag_model_list)
# print("轉推數高中低model選這些: \n",retweet_count_model_list)
# print("轉推時間分布model選這些: \n",distribution_model_list)
#----------------------------------------將所有要跑結果的data graph執行------------------------------------------------------------------------
for number in data_graph_list:
    #----------------------------------------將轉推時間分佈讀取出來----------------------------------------------------------------------------------------------------
    retweet_distribution = np.load(elon_mask_base_graph_data_dir + 'retweet_distribution/'+
                                   str(number[0]) + '_graph_retweet_distribution_rewrite.npy')
    for time_point in range(10):
        retweet_distribution_mask = np.ones(retweet_distribution.shape[0])
        retweet_distribution_mask = retweet_distribution_mask - retweet_distribution[:,time_point]
        retweet_distribution_mask = th.from_numpy(retweet_distribution_mask).type(th.bool)
        print('*****************************************************************************************************************************************************************************************')
        # print("graph number: ",number[0])
        # print("now graph's time point: ",time_point)
        # print("graph tagt's tag: ",number[1])
        # print('graph\'s retweet distribution: \n',list(retweet_distribution[:,time_point]))
        # print('graph\'s retweet distribution mask: \n',list(retweet_distribution_mask))
        #------------------------------------------------------------------------------------------------------------------------
        #tag 的結果
        if int(number[1]) != 1:
            dataset,label = load_graphs(data_graph_dir + str(number[0]) + "/" + 
                                        str(time_point) + "/" + str(number[0]) + "_rewrite.bin")
            Graph = dataset[0]
            cuda_g = Graph.to('cuda:0')
            features = cuda_g.ndata['feature']
            labels = cuda_g.ndata['label']
            net = Net(len(features[0]))
            # print("model number is: ",str(tag_model_list[np.where(tag_model_list[:,0] == (number[1]))[0][0]][1]))
            # print("the result round of model is: ",str(tag_model_list[np.where(tag_model_list[:,0] == (number[1]))[0][0]][2]))
            net.load_state_dict(th.load(model_choose_file_dir+str(tag_model_list[np.where(tag_model_list[:,0] == (number[1]))[0][0]][1])+'/'+str(tag_model_list[np.where(tag_model_list[:,0] == (number[1]))[0][0]][2])+'_round_result_edge_weight_Adjustment.pth'))
            tag_test = model_test()
            print('tag\'s acc is: {:.6f}'.format(tag_test.test_model(net,cuda_g,features,labels,retweet_distribution_mask)))
            tag_test.model_save(data_graph_dir + str(number[0]) + "/" +
                                str(time_point) + "/" + str(number[0]) + '_tag_model_result')
            del net
            print('-------------------------------------------------------------分隔線-------------------------------------------------------------------------')
        #------------------------------------------------------------------------------------------------------
        #retweet count的結果
            # print('model for high retweet count is: ',retweet_count_model_list[0][1])
            # print('round of model for high retweet count is: ',retweet_count_model_list[0][2])
            net = Net(len(features[0]))
            net.load_state_dict(th.load(model_choose_file_dir + str(retweet_count_model_list[0][1]) +
                                        '/' + str(retweet_count_model_list[0][2]) + '_round_result_edge_weight_Adjustment.pth'))
            retweet_count_high = model_test()
            print('model of high retweet count acc is: {:.6f}'.format(retweet_count_high.test_model(net,cuda_g,features,labels,retweet_distribution_mask)))
            retweet_count_high.model_save(data_graph_dir+str(number[0]) + "/" + str(time_point) + "/" + str(number[0]) +
                                          '_high_retweet_count_model_result')
            del net

            # print('model for middle retweet count is: ',retweet_count_model_list[1][1])
            # print('round of model for middle retweet count is: ',retweet_count_model_list[1][2])
            net = Net(len(features[0]))
            net.load_state_dict(th.load(model_choose_file_dir + str(retweet_count_model_list[1][1]) + '/' +
                                        str(retweet_count_model_list[1][2]) + '_round_result_edge_weight_Adjustment.pth'))
            retweet_count_middle = model_test()
            print('model of middle retweet count acc is: {:.6f}'.format(retweet_count_middle.test_model(net,cuda_g,features,labels,retweet_distribution_mask)))
            retweet_count_middle.model_save(data_graph_dir + str(number[0]) + "/" + str(time_point) +
                                            "/" + str(number[0]) + '_middle_retweet_count_model_result')
            del net

            # print('model for low retweet count is: ',retweet_count_model_list[2][1])
            # print('round of model for low retweet count is: ',retweet_count_model_list[2][2])
            net = Net(len(features[0]))
            net.load_state_dict(th.load(model_choose_file_dir+str(retweet_count_model_list[2][1]) + '/' + 
                                        str(retweet_count_model_list[2][2]) + '_round_result_edge_weight_Adjustment.pth'))
            retweet_count_low = model_test()
            print('model of low retweet count acc is: {:.6f}'.format(retweet_count_low.test_model(net,cuda_g,features,labels,retweet_distribution_mask)))
            retweet_count_low.model_save(data_graph_dir + str(number[0]) + "/" + str(time_point) + 
                                         "/" + str(number[0]) + '_low_retweet_count_model_result')
            del net
        #------------------------------------------------------------------------------------------------------
        #是tag 3的特別處裡
            if int(number[1]) == 3:
                print('-------------------------------------------------------------分隔線-------------------------------------------------------------------------')
                # print('model for longest time collected is: ',distribution_model_list[0][1])
                # print('round of model for longest time collected is: ',distribution_model_list[0][2])
                net = Net(len(features[0]))
                net.load_state_dict(th.load(model_choose_file_dir+str(distribution_model_list[0][1]) + '/'+ 
                                            str(distribution_model_list[0][2]) + '_round_result_edge_weight_Adjustment.pth'))
                retweet_longest = model_test()
                print('model of longest time acc is: {:.6f}'.format(retweet_longest.test_model(net,cuda_g,features,labels,retweet_distribution_mask)))
                retweet_longest.model_save(data_graph_dir+str(number[0]) + "/" + str(time_point) + 
                                           "/" + str(number[0]) + '_over_72_model_result')
                del net

                # print('model for longer time collected is: ',distribution_model_list[1][1])
                # print('round of model for longer time collected is: ',distribution_model_list[1][2])
                net = Net(len(features[0]))
                net.load_state_dict(th.load(model_choose_file_dir+str(distribution_model_list[1][1]) + '/' + 
                                            str(distribution_model_list[1][2]) + '_round_result_edge_weight_Adjustment.pth'))
                retweet_longer = model_test()
                print('model of longer time acc is: {:.6f}'.format(retweet_longer.test_model(net,cuda_g,features,labels,retweet_distribution_mask)))
                retweet_longer.model_save(data_graph_dir+str(number[0]) + "/" + 
                                          str(time_point) + "/" + str(number[0]) + '_between_48_72_model_result')
                del net

                # print('model for middle time collected is: ',distribution_model_list[2][1])
                # print('round of model for middle time collected is: ',distribution_model_list[2][2])
                net = Net(len(features[0]))
                net.load_state_dict(th.load(model_choose_file_dir+str(distribution_model_list[2][1]) + '/' + 
                                            str(distribution_model_list[2][2]) + '_round_result_edge_weight_Adjustment.pth'))
                retweet_middle = model_test()
                print('model of middle time acc is: {:.6f}'.format(retweet_middle.test_model(net,cuda_g,features,labels,retweet_distribution_mask)))
                retweet_middle.model_save(data_graph_dir + str(number[0]) + "/" + str(time_point) +
                                          "/" + str(number[0]) + '_between_24_48_model_result')
                del net

                # print('model for shorter time collected is: ',distribution_model_list[3][1])
                # print('round of model for shorter time collected is: ',distribution_model_list[3][2])
                net = Net(len(features[0]))
                net.load_state_dict(th.load(model_choose_file_dir+str(distribution_model_list[3][1]) + '/' + 
                                            str(distribution_model_list[3][2]) + '_round_result_edge_weight_Adjustment.pth'))
                retweet_shorter = model_test()
                print('model of shorter time acc is: {:.6f}'.format(retweet_shorter.test_model(net,cuda_g,features,labels,retweet_distribution_mask)))
                retweet_shorter.model_save(data_graph_dir+str(number[0]) + "/" + str(time_point) + "/" + 
                                           str(number[0]) + '_between_12_24_model_result')
                del net

                # print('model for shortest time collected is: ',distribution_model_list[4][1])
                # print('round of model for shortest time collected is: ',distribution_model_list[4][2])
                net = Net(len(features[0]))
                net.load_state_dict(th.load(model_choose_file_dir+str(distribution_model_list[4][1]) + '/' + 
                                            str(distribution_model_list[4][2]) + '_round_result_edge_weight_Adjustment.pth'))
                retweet_shortest = model_test()
                print('model of shortest time acc is: {:.6f}'.format(retweet_shortest.test_model(net,cuda_g,features,labels,retweet_distribution_mask)))
                retweet_shortest.model_save(data_graph_dir + str(number[0]) + "/" + 
                                            str(time_point) + "/" + str(number[0]) + '_in_12_model_result')
                del net
        # print('*****************************************************************************************************************************************************************************************')

end = time.time()

total_time = end - start
sec = int(total_time % 60)
min = int(total_time / 60)
print("執行時間：%d分 %d 秒" %(min, sec))
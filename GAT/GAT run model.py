import os
os.environ["DGLBACKEND"] = "pytorch"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import time
from GAT_No_edge_weight_model_run import GAT_model_run
# from GAT_model_run import GAT_model_run
# from GATv2Conv_model import GATv2_model_run

start = time.time()
base_graph_for_model_dir = 'D:/GCN_Twitter/ElonMusk/2023-02-16/base_graph_for_model/'


for number_graph in range(0, 1, 1):#27    
    for batch in range(5):
        print('GAT(No edge weight) now run graph nubmer: {},round: {}'.format(number_graph, batch))
        model_construct = GAT_model_run()
        # load 未加權圖
        model_construct.elonmask_dataset_load(number_graph, batch)
        # GAT module(no weight) run && store
        model_construct.GAT_run_No_edge_weight_model()
        model_construct.GAT_model_save(base_graph_for_model_dir + 'GAT_FocalLoss1013/' + str(number_graph) + '/' + 
                                       str(batch) + '_round_result_GAT_FocalLoss', batch) 
        
        # model_construct2 = GATv2_model_run()
        # model_construct2.elonmask_dataset_load(number_graph, batch)
        # # GAT module(no weight) run && store
        # model_construct2.GATv2_run_model()
        # model_construct2.GAT_model_save(base_graph_for_model_dir + 'GAT_graph_no_weight_model2/' + str(number_graph) + '/' + 
        #                                str(batch) + '_round_result_No_edge_weight_Adjustment_GATv2(test)', batch)         

end = time.time()
total_time = end - start
min = int(total_time / 60)
hr = int(min / 60)
sec = int(total_time % 60)
min -= hr * 60
print("執行時間：%d小時 %d分 %d 秒" %(hr, min, sec))



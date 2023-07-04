import sys
sys.path.append("D:/GCN_Twitter/GCN_code/gcn")
import time
from model_run import model_run

start = time.time()
base_graph_for_model_dir = 'D:/GCN_Twitter/ElonMusk/2023-02-16/base_graph_for_model/'
for number_graph in range(0,1,1):#27
    for batch in range(1):#5
        print('now run graph nubmer: {},round: {}'.format(number_graph,batch))
        model_construct = model_run()
        model_construct.elonmask_dataset_load(number_graph, batch)
        # GCN module run && store
        # model_construct.run_model()
        # model_construct.model_save(base_graph_for_model_dir + str(number_graph) + '/' +
        #                             str(batch) + '_round_result_edge_weight_Adjustment', batch)
        # GAT module run && store
        model_construct.GAT_run_model()
        # model_construct.GAT_model_save(base_graph_for_model_dir + str(number_graph) + '/' +
        #                             str(batch) + '_round_result_edge_weight_Adjustment_GAT', batch) 


end = time.time()

total_time = end - start
min = int(total_time / 60)
hr = int(min / 60)
sec = int(total_time % 60)
min -= hr * 60
print("執行時間：%d小時 %d分 %d 秒" %(hr, min, sec))
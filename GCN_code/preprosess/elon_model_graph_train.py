import sys
sys.path.append("D:/GCN_Twitter/GCN_code/gcn")
import time
from model_run import model_run

start = time.time()
base_graph_for_model_dir = 'D:/GCN_Twitter/ElonMusk/2023-02-16/base_graph_for_model/'
for number_graph in range(0,27,1):#27
    for batch in range(5):#5
        print('now run graph nubmer: {},round: {}'.format(number_graph,batch))
        model_construct = model_run()
        model_construct.elonmask_dataset_load(number_graph,batch)
        model_construct.run_model()
        model_construct.model_save(base_graph_for_model_dir + str(number_graph) + '/' +
                                    str(batch) + '_round_result_edge_weight_Adjustment',batch) 

end = time.time()

total_time = end - start
sec = int(total_time % 60)
min = int(total_time / 60)
print("執行時間：%d分 %d 秒" %(min, sec))
import sys
sys.path.append("D:/GNN/code/gcn")
from model_run import model_run

for number_graph in range(0,27,1):#27
    for batch in range(5):#5
        print('now run graph nubmer: {},round: {}'.format(number_graph,batch))
        model_construct = model_run()
        model_construct.elonmask_dataset_load(number_graph,batch)
        model_construct.run_model()
        model_construct.model_save('D:/GNN/MyResearch-main/MyResearch-main/ElonMusk/2023-02-16/base_graph_for_model/'+str(number_graph)+'/'+str(batch)+'_round_result_edge_weight_Adjustment',batch) 
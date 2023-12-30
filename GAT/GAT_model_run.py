import sys
sys.path.append("D:/GCN_Twitter/GCN_code/gcn")
import time
import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from load_GAT_database import load_database 
import dgl
import json 
from balanced_loss import Loss
import matplotlib.pyplot as plt
import os
from gat import GAT
#跨資料夾引用
from evaluate import model_evaluate

base_graph_for_model_dir = 'D:/GCN_Twitter/ElonMusk/2023-02-16/base_graph_for_model/'
class GAT_model_run():
    database = load_database()
    Epoch_GAT = 4000
    confusionmatrix = list()

    def elonmask_dataset_load(self, number, round):
        self.graph, self.features, self.labels, self.train_mask, self.test_mask = self.database.load_elon_dataset(number,round)
        # Add edges between each node and itself to preserve old node representations
        # self.graph.add_edges(self.graph.nodes(), self.graph.nodes())         #add_edges   從src_node -> dst_node    add_edge(src_node_list,dst_node_list)  

    def elonmask_no_weight_dataset_load(self, number, round):
        self.graph, self.features, self.labels, self.train_mask, self.test_mask = self.database.load_elon_no_weight_dataset(number,round)
        # Add edges between each node and itself to preserve old node representations
        # self.graph.add_edges(self.graph.nodes(), self.graph.nodes())   
         
    def draw_GAT_Loss(self, PATH, loss_list, acc_list):
        for graph_num in range(len(os.listdir(PATH)) + 1):
            Epoch = [i for i in range(1, self.Epoch_GAT + 1)]
            if not os.path.isfile(PATH + str(graph_num) + '_GAT_Loss0712.png'):    
                # print('make picture: ', gat_loss_list)        
                plt.plot(Epoch, loss_list, label = "loss") 
                plt.plot(Epoch, acc_list, label = 'Acc')  
                plt.xlabel("epoch")
                plt.legend(
                    loc = 'best',
                    fontsize = 10,
                    shadow = False,)
                plt.title('GAT_Loss', loc='center')
                plt.savefig(PATH + str(graph_num) + '_GAT_Loss0712.png')
                plt.close()

    def GAT_run_model(self):
        # print(self.features.size()[1])   #feature = 200 + 4
        print('GAT: ', self.graph)
        self.net = GAT( 
        g = self.graph,
        in_dim = self.features.size()[1], 
        hidden_dim = 8, # hidden layer dimension
        out_dim = 2, # output dimension, equal to number of class
        num_heads = 2 # multi-head attention
        )

        optimizer = th.optim.Adam(self.net.parameters(), lr = 1e-3)
        dur = []
        gat_loss_list = []
        acc_list = []
        path = 'D:/GCN_Twitter/ElonMusk/2023-02-16/GAT_graph2/'
        for epoch in range(self.Epoch_GAT):
            if epoch >=3:
                t0 = time.time()

            gat_logits = self.net(self.graph, self.features.float())
            # print(logits)
            gat_logp = F.log_softmax(gat_logits, 1)     # shape = |V| * classes
            gat_loss = F.nll_loss(gat_logp[self.train_mask], self.labels[self.train_mask])

            optimizer.zero_grad()
            gat_loss.backward()
            optimizer.step()
            
            #Focal Loss
            # focal_loss = Loss(loss_type = 'focal_loss')
            # loss = focal_loss(logits[self.train_mask], self.labels[self.train_mask])
            if epoch >= 3:
                dur.append(time.time() - t0)
            
            gat_test = model_evaluate()
            acc = gat_test.evaluate(self.net, self.graph, self.features, self.labels, self.test_mask)
            self.confusionmatrix = gat_test.counfusionmatrix
            print("GAT：Epoch {:05d} | Loss {:.4f} | Test Acc {:.4f} | Time(s) {:.4f}".format(
                    epoch, gat_loss.item(), acc, np.mean(dur)))
            gat_loss_list.append(gat_loss.item())
            acc_list.append(acc)
            # print(gat_loss_list)  
        # 畫loss圖          
        self.draw_GAT_Loss(path, gat_loss_list, acc_list)

    def GAT_model_save(self,PATH,round):
        print('----------------------------------------------------------------------------------------------')                       
        evaluate_value = dict()
        evaluate_value['TP'] = int(self.confusionmatrix[1][1])                   #   T   F   
        evaluate_value['FP'] = int(self.confusionmatrix[0][1])                   #N  TN  FP
        evaluate_value['FN'] = int(self.confusionmatrix[1][0])                   #P  FN  TP
        evaluate_value['TN'] = int(self.confusionmatrix[0][0])
        try:
            evaluate_value['Accuracy'] = (evaluate_value['TP'] + evaluate_value['TN']) / (evaluate_value['TP'] +  evaluate_value['FP'] + evaluate_value['FN'] + evaluate_value['TN'])
        except:
            evaluate_value['Accuracy'] = 0
        try:
            evaluate_value['Precision'] = evaluate_value['TP'] / (evaluate_value['TP'] + evaluate_value['FP'])
        except:
            evaluate_value['Precision'] = 0
        try:
            evaluate_value['Recall'] = evaluate_value['TP'] / (evaluate_value['TP'] + evaluate_value['FN'])
        except:
            evaluate_value['Recall'] = 0
        try:
            evaluate_value['F1'] = 2 / ((1 / evaluate_value['Precision']) + (1 / evaluate_value['Recall']))
        except:
            evaluate_value['F1'] = 0
        th.save(self.net.state_dict(), PATH + '.pth')
        with open(PATH + ".json", "w") as outfile:
            json.dump(evaluate_value, outfile)    
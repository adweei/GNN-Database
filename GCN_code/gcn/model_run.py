import time
import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from gcn_model import Net 
from load_database import load_database 
from evaluate import model_evaluate
import dgl
import json 

class model_run():
    database = load_database()
    Epoch = 100
    net = Net(1)
    confusionmatrix = list()

    def coradataset_load(self):
        self.graph, self.features, self.labels, self.train_mask, self.test_mask = self.database.load_cora_data()
        # Add edges between each node and itself to preserve old node representations
        self.graph.add_edges(self.graph.nodes(), self.graph.nodes())

    def whitehosedataset_load(self,round):
        self.graph, self.features, self.labels = self.database.load_whitehose_dataset()
        self.train_mask ,self.test_mask = self.database.load_selfdefinemask(round)
        self.graph.add_edges(self.graph.nodes(), self.graph.nodes())

    def elonmask_dataset_load(self,number,round):
        self.graph, self.features, self.labels, self.train_mask, self.test_mask = self.database.load_elon_dataset(number,round)
        # Add edges between each node and itself to preserve old node representations
        self.graph.add_edges(self.graph.nodes(), self.graph.nodes())

    def run_model(self):
        self.net = Net(len(self.features[0]))
        optimizer = th.optim.Adam(self.net.parameters(), lr=1e-2)
        dur = []
        for epoch in range(self.Epoch):
            if epoch >=3:
                t0 = time.time()

            self.net.train()
            logits = self.net(self.graph, self.features)
            logp = F.log_softmax(logits, 1)
            loss = F.nll_loss(logp[self.train_mask], self.labels[self.train_mask])
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if epoch >=3:
                dur.append(time.time() - t0)
            
            test = model_evaluate()
            acc = test.evaluate(self.net, self.graph, self.features, self.labels, self.test_mask)
            self.confusionmatrix = test.counfusionmatrix
            print("Epoch {:05d} | Loss {:.4f} | Test Acc {:.4f} | Time(s) {:.4f}".format(
                    epoch, loss.item(), acc, np.mean(dur)))
        
    def model_save(self,PATH,round):
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
        th.save(self.net.state_dict(), PATH+'.pth')
        with open(PATH+".json", "w") as outfile:
            json.dump(evaluate_value, outfile)


# a = model_run()
# a.whitehosedataset_load(3)
# a.run_model()

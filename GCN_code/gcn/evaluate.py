import torch as th
from torchmetrics.classification import BinaryConfusionMatrix

class model_evaluate():
    counfusionmatrix = list()

    def evaluate(self,model, g, features, label, mask):
        model.eval()
        with th.no_grad():
            logits = model(g, features)
            logits = logits[mask]
            labels = label[mask]
            _, indices = th.max(logits, dim=1)
            #cuda_indices = indices.to('cuda:0')
            correct = th.sum(indices == labels)
            bcm = BinaryConfusionMatrix().to(device = 'cuda:0')#cpu cuda:0
            self.counfusionmatrix = bcm(indices, labels)

            return correct.item() * 1.0 / len(labels)
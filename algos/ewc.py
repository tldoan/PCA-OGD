from copy import deepcopy
import torch.nn.functional as F
## toolbox for EWC

def consolidate(trainer,precision_matrices):
   for index in trainer._precision_matrices:
       trainer._precision_matrices[index]+=precision_matrices[index]
             
def update_means(trainer):

    for n, p in deepcopy(trainer.params).items():
        trainer._means[n] = p.data
        
def penalty(trainer):
    loss = 0
    
    for index in trainer.params:
        _loss = trainer._precision_matrices[index] *0.5* (trainer.params[index] - trainer._means[index]) ** 2
        loss += _loss.sum()
    return loss
        
def _diag_fisher(trainer,train_loader):
    precision_matrices = {}
    
    for n, p in deepcopy(trainer.params).items():
        p.data.zero_()
        precision_matrices[n] = p.data

    trainer.model.eval()
    k=1
    for _,element in enumerate(train_loader):
        if k>=trainer.config.fisher_sample:
            break
        trainer.model.zero_grad()
        inputs=element[0].to(trainer.config.device)
        targets = element[1].long().to(trainer.config.device)
       
        task=element[2]
        out = trainer.forward(inputs,task)
        assert out.shape[0] == 1
      
        pred = out.cpu()

       
        loss=F.log_softmax(pred, dim=1)[0][targets.item()]
        loss.backward()

        for index in trainer.params:
            trainer._precision_matrices[index].data += trainer.params[index].grad.data ** 2 / trainer.config.fisher_sample
        
    precision_matrices = {n: p for n, p in precision_matrices.items()}
    return precision_matrices
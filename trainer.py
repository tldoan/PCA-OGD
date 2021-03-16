from torch.nn.utils.convert_parameters import  parameters_to_vector, vector_to_parameters
import torch.nn as nn
import torch
from collections import defaultdict


from types import MethodType
from importlib import import_module

from utils.utils import parameters_to_grad_vector,count_parameter
from copy import deepcopy
from algos import gem,ogd
        
        


class Trainer(object):
    def __init__(self, config,val_loaders):
        self.config=config
        
       
        self.model = self.create_model()
       
        self.optimizer = torch.optim.SGD(params=self.model.parameters(),lr=self.config.lr,momentum=0,weight_decay=0)
        self.criterion = nn.CrossEntropyLoss()
        
        
        
        n_params = count_parameter(self.model)
        self.ogd_basis = torch.empty(n_params, 0).to(self.config.device)
       
   
        self.mem_dataset = None

   
        self.ogd_basis_ids = defaultdict(lambda: torch.LongTensor([]).to(self.config.device))

        self.val_loaders = val_loaders

        self.task_count = 0
        self.task_memory = {}
        
       
        ### FOR GEM no transfer
        self.task_mem_cache = {}

        self.task_grad_memory = {}
        self.task_grad_mem_cache = {}
        
        ### Creating dictionary to save accuracy / can also save loss functions but have not implemented it
        self.acc={}
        for element in range(self.config.n_tasks):
            self.acc[element]={}
            self.acc[element]['test_acc']=[]
            self.acc[element]['training_acc']=[]
            self.acc[element]['training_steps']=[]
        
        if self.config.method=="gem-nt":
            self.quadprog = import_module('quadprog')
            self.grad_to_be_saved={}
        if self.config.method=="agem":
            self.agem_mem = list()
            self.agem_mem_loader = None
            
        
        self.gradient_count=0
        self.gradient_violation=0
        self.eval_freq=config.eval_freq
        
        if self.config.method=="ewc":
            ## split cifar
            if self.config.is_split:
                if self.config.all_features:
                    if hasattr(self.model,"conv"):
                        r=list(self.model.linear.named_parameters())+list(self.model.conv.named_parameters())
                        self.params = {n: p for n, p in r if p.requires_grad}
                    else:
                        self.params = {n: p for n, p in self.model.linear.named_parameters() if p.requires_grad}
                else:
                        self.params = {n: p for n, p in self.model.linear.named_parameters() if p.requires_grad}
                   
            ### rotated
            else:
                self.params = {n: p for n, p in self.model.named_parameters() if p.requires_grad}
            
            self._means = {}
            
            for n, p in deepcopy(self.params).items():
                self._means[n] = p.data
            
            self._precision_matrices ={}
            for n, p in deepcopy(self.params).items():
                p.data.zero_()
                self._precision_matrices[n] = p.data
    
    
     
        
    def create_model(self):
      
        cfg = self.config
        
        
        if "cifar" not in cfg.dataset:
            import models.mlp
            
            model = models.mlp.MLP(hidden_dim=cfg.hidden_dim)
        
        else:
            import models.lenet
            
            model = models.lenet.LeNetC(hidden_dim=cfg.hidden_dim)
      
                                                                             
        n_feat = model.last.in_features
       
        model.last = nn.ModuleDict()
        for task,out_dim in cfg.out_dim.items():
           
            model.last[task] = nn.Linear(n_feat,out_dim)

      

        # Redefine the task-dependent function
        def new_logits(self, x):
            outputs = {}
            for task, func in self.last.items():
                outputs[task] = func(x)
            return outputs

        # Replace the task-dependent function
        model.logits = MethodType(new_logits, model)
        model.to(self.config.device)
        return model

    def forward(self, x, task):
       
        task_key = task[0]
        out = self.model.forward(x)
        # print(out)
        if self.config.is_split :
            try:
                return out[task_key]
            except:
                return out[int(task_key)]
        else :
            # return out
            return out["All"] 
    
    def get_params_dict(self, last, task_key=None):
        if self.config.is_split :
            if last:
                return self.model.last[task_key].parameters()
            else:
              
                if self.config.all_features:
                    ## take the conv parameters into account
                    if hasattr(self.model,"conv"):
                        return list(self.model.linear.parameters())+list(self.model.conv.parameters())
                    else:
                        return self.model.linear.parameters()
                    
                        
                else:
                    return self.model.linear.parameters()
                # return self.model.linear.parameters()
        else:
            return self.model.parameters()
        
    

    
    
    def optimizer_step(self):
        
        task_key = str(self.task_id)
        
        ### take gradients with respect to the parameters
        grad_vec = parameters_to_grad_vector(self.get_params_dict(last=False))
        cur_param = parameters_to_vector(self.get_params_dict(last=False))
        
        if self.config.method in ['ogd','pca']:
       
            proj_grad_vec = ogd.project_vec(grad_vec,
                                        proj_basis=self.ogd_basis)
            ## take the orthogonal projection
            new_grad_vec = grad_vec - proj_grad_vec
           
        elif self.config.method=="agem" and self.agem_mem_loader is not None :
            self.optimizer.zero_grad()
            data, target, task = next(iter(self.agem_mem_loader))
            # data = self.to_device(data)
            data=data.to(self.config.device)
            target = target.long().to(self.config.device)
          
            output = self.forward(data, task)
            mem_loss = self.criterion(output, target)
            mem_loss.backward()
            mem_grad_vec = parameters_to_grad_vector(self.get_params_dict(last=False))
            
            self.gradient_count+=1
              
           
            new_grad_vec = gem._project_agem_grad(self,batch_grad_vec=grad_vec,
                                                   mem_grad_vec=mem_grad_vec)
        elif self.config.method=="gem-nt":
         
            if self.task_count >= 1:
                
                 for t,mem in self.task_memory.items():
                    self.optimizer.zero_grad()
                   
                    mem_out = self.forward(self.task_mem_cache[t]['data'].to(self.config.device),self.task_mem_cache[t]['task'])
                    
                    mem_loss = self.criterion(mem_out, self.task_mem_cache[t]['target'].long().to(self.config.device))
               
                    mem_loss.backward()
                  
                    self.task_grad_memory[t]=parameters_to_grad_vector(self.get_params_dict(last=False))
                   
                    mem_grad_vec = torch.stack(list(self.task_grad_memory.values()))
                   
                    new_grad_vec = gem.project2cone2(self,grad_vec, mem_grad_vec)
                    
            else:
                new_grad_vec = grad_vec
            
        else:
            new_grad_vec = grad_vec
            
        ### SGD update  =>  new_theta= old_theta - learning_rate x ( derivative of loss function wrt parameters )
        cur_param -= self.config.lr * new_grad_vec#.to(self.config.device)
        
        vector_to_parameters(cur_param, self.get_params_dict(last=False))
     
        if self.config.is_split :
            # Update the parameters of the last layer without projection, when there are multiple heads)
            cur_param = parameters_to_vector(self.get_params_dict(last=True, task_key=task_key))
            grad_vec = parameters_to_grad_vector(self.get_params_dict(last=True, task_key=task_key))
            cur_param -= self.config.lr * grad_vec
            vector_to_parameters(cur_param, self.get_params_dict(last=True, task_key=task_key))
        
        ### zero grad
        self.optimizer.zero_grad()
        
  


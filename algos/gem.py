import torch
from utils.utils import parameters_to_grad_vector
from algos.common import Memory
import numpy as np
## toolbox for GEM/AGEM methods

def _get_new_gem_m_basis(self, device,optimizer, model,forward):
        new_basis = []
    
        
      
        for t,mem in self.task_memory.items():
          
         
            for index in range(len(self.task_mem_cache[t]['data'])):

            
                inputs=self.task_mem_cache[t]['data'][index].to(device).unsqueeze(0)
                
                targets=self.task_mem_cache[t]['target'][index].to(device).unsqueeze(0)
                
                task=self.task_mem_cache[t]['task'][0]
                
                out = forward(inputs,task)
               
                mem_loss = self.criterion(out, targets.long().to(self.config.device))
                optimizer.zero_grad()
                mem_loss.backward()
                new_basis.append(parameters_to_grad_vector(self.get_params_dict(last=False)).cpu())
            
        del out,inputs,targets
        torch.cuda.empty_cache()
        new_basis = torch.stack(new_basis).T
        return new_basis
    
    
def update_agem_memory(trainer, train_loader,task_id):
        
        trainer.task_count =task_id
        num_sample_per_task = trainer.config.memory_size #// (self.config.n_tasks-1)
        randind = torch.randperm(len(train_loader.dataset))[:num_sample_per_task]
        for ind in randind:  # save it to the memory
            trainer.agem_mem.append(train_loader.dataset[ind])
            
    
        
        mem_loader_batch_size = min(trainer.config.agem_mem_batch_size, len(trainer.agem_mem))
        trainer.agem_mem_loader = torch.utils.data.DataLoader(trainer.agem_mem,
                                                   batch_size=mem_loader_batch_size,
                                                   shuffle=True,
                                                   num_workers=1)
        
        
def update_gem_no_transfer_memory(trainer,train_loader,task_id):
        
        trainer.task_count=task_id
       
        num_sample_per_task = trainer.config.memory_size 
        
        num_sample_per_task = min(len(train_loader.dataset), num_sample_per_task)
     
        trainer.task_memory[trainer.task_count] = Memory()
        randind = torch.randperm(len(train_loader.dataset))[:num_sample_per_task]  # randomly sample some data
        for ind in randind:  # save it to the memory
            trainer.task_memory[trainer.task_count].append(train_loader.dataset[ind])
            
      
        for t, mem in trainer.task_memory.items():
          
            mem_loader = torch.utils.data.DataLoader(mem,
                                                     batch_size=len(mem),
                                                     shuffle=False,
                                                     num_workers=1)
            assert len(mem_loader) == 1, 'The length of mem_loader should be 1'
            for i, (mem_input, mem_target, mem_task) in enumerate(mem_loader):
                pass
              
            trainer.task_mem_cache[t] = {'data': mem_input, 'target': mem_target, 'task': mem_task}
            
            


        
def _project_agem_grad(trainer, batch_grad_vec, mem_grad_vec):
   
        if torch.dot(batch_grad_vec, mem_grad_vec) >= 0:
            return batch_grad_vec
        else :
            trainer.gradient_violation+=1
          
            
          
            frac = torch.dot(batch_grad_vec, mem_grad_vec) / torch.dot(mem_grad_vec, mem_grad_vec)
           
            new_grad = batch_grad_vec - frac * mem_grad_vec
           
            check = torch.dot(new_grad, mem_grad_vec)
            assert torch.abs(check) < 1e-5
            return new_grad

def project2cone2(trainer, gradient, memories):
        """
            Solves the GEM dual QP described in the paper given a proposed
            gradient "gradient", and a memory of task gradients "memories".
            Overwrites "gradient" with the final projected update.
            input:  gradient, p-vector
            input:  memories, (t * p)-vector
            output: x, p-vector
            Modified from: https://github.com/facebookresearch/GradientEpisodicMemory/blob/master/model/gem.py#L70
        """
       
        margin = trainer.config.margin_gem
        memories_np = memories.cpu().contiguous().double().numpy()
        gradient_np = gradient.cpu().contiguous().view(-1).double().numpy()
        t = memories_np.shape[0]
       
        P = np.dot(memories_np, memories_np.transpose())
        P = 0.5 * (P + P.transpose())
        q = np.dot(memories_np, gradient_np) * -1
        G = np.eye(t)
        P = P + G * 0.001
        h = np.zeros(t) + margin
        # print(P)
        v = trainer.quadprog.solve_qp(P, q, G, h)[0]
        x = np.dot(v, memories_np) + gradient_np
        new_grad = torch.Tensor(x).view(-1)
       
        new_grad = new_grad.to(trainer.config.device)
        return new_grad
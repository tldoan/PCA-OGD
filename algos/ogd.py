import torch
from algos.common import Memory
from utils.utils import parameters_to_grad_vector, count_parameter

def _get_new_ogd_basis(trainer,train_loader, device,optimizer, model,forward):
        new_basis = []
    
        
        for _,element in enumerate(train_loader):

            inputs=element[0].to(device)
          
            targets = element[1].to(device)
            
            task=element[2]
            
            out = forward(inputs,task)
          
            assert out.shape[0] == 1
          
            pred = out[0,int(targets.item())].cpu()
          
            optimizer.zero_grad()
            pred.backward()
        
            ### retrieve  \nabla f(x) wrt theta
            new_basis.append(parameters_to_grad_vector(trainer.get_params_dict(last=False)).cpu())
         
        del out,inputs,targets
        torch.cuda.empty_cache()
        new_basis = torch.stack(new_basis).T
      
        return new_basis
    
def project_vec(vec, proj_basis):
        if proj_basis.shape[1] > 0 :  # param x basis_size
            dots = torch.matmul(vec, proj_basis)  # basis_size  dots= [  <vec, i >   for i in proj_basis ]
            out = torch.matmul(proj_basis, dots)  # out = [  <vec, i > i for i in proj_basis ]
            return out
        else:
            return torch.zeros_like(vec) 
    
def update_mem(trainer,train_loader,task_count):
        
        trainer.task_count =task_count
    
      
        num_sample_per_task = trainer.config.memory_size # // (self.config.n_tasks-1)
        num_sample_per_task = min(len(train_loader.dataset), num_sample_per_task)
    
        memory_length=[]
        for i in range(task_count):
             memory_length.append(num_sample_per_task)
      
        
        for storage in trainer.task_memory.values():
            ## reduce the size of the stored elements
            storage.reduce(num_sample_per_task)
           
       
        trainer.task_memory[0] = Memory()  # Initialize the memory slot
      
        if trainer.config.method=="pca":
            randind = torch.randperm(len(train_loader.dataset))[:trainer.config.pca_sample]  ## for pca method we  samples pca_samples > num_sample_per_task before applying pca and keeping (num_sample_per_task) elements
        else:
            randind = torch.randperm(len(train_loader.dataset))[:num_sample_per_task]  # randomly sample some data
        for ind in randind:  # save it to the memory
          
            trainer.task_memory[0].append(train_loader.dataset[ind])


        ####################################### Grads MEM ###########################
      
        for storage in trainer.task_grad_memory.values():
            storage.reduce(num_sample_per_task)
    

       
        if trainer.config.method in ['ogd','pca']:
            ogd_train_loader = torch.utils.data.DataLoader(trainer.task_memory[0],
                                                       batch_size=1,
                                                       shuffle=False,
                                                       num_workers=1)
          
        
        
        trainer.task_memory[0] = Memory()
      
        new_basis_tensor = _get_new_ogd_basis(trainer,
                                              ogd_train_loader,
                                              trainer.config.device,
                                              trainer.optimizer,
                                              trainer.model,
                                              trainer.forward).cpu()
        
      
                    
        if trainer.config.method=="pca":
          
            try:
                _,_,v1=torch.pca_lowrank(new_basis_tensor.T.cpu(), q=num_sample_per_task, center=True, niter=2)
               
            except:
                _,_,v1=torch.svd_lowrank((new_basis_tensor.T+1e-4*new_basis_tensor.T.mean()*torch.rand(new_basis_tensor.T.size(0), new_basis_tensor.T.size(1))).cpu(), q=num_sample_per_task, niter=2, M=None)
            
          
                
            del new_basis_tensor
            new_basis_tensor=v1.cpu()
            torch.cuda.empty_cache()
          
        if trainer.config.is_split:
            if trainer.config.all_features:
                if hasattr(trainer.model,"conv"):
                        n_params = count_parameter(trainer.model.linear)+count_parameter(trainer.model.conv)
                else:
                        n_params = count_parameter(trainer.model.linear)
            else:
                
                n_params = count_parameter(trainer.model.linear)
               
        else:
            n_params = count_parameter(trainer.model)
            
        
     
        trainer.ogd_basis = torch.empty(n_params, 0).cpu()
      
      
        for t, mem in trainer.task_grad_memory.items():
         
            task_ogd_basis_tensor=torch.stack(mem.storage,axis=1).cpu()
           
            trainer.ogd_basis = torch.cat([trainer.ogd_basis, task_ogd_basis_tensor], axis=1).cpu()
          
      
        trainer.ogd_basis=orthonormalize(trainer.ogd_basis,new_basis_tensor,trainer.config.device,normalize=True)
      
    
        # (g) Store in the new basis
        ptr = 0
     
        for t in range(len(memory_length)):
            
            
            task_mem_size=memory_length[t]
            idxs_list = [i + ptr for i in range(task_mem_size)]
           
            trainer.ogd_basis_ids[t] = torch.LongTensor(idxs_list).to(trainer.config.device)
            
           
            trainer.task_grad_memory[t] = Memory()  # Initialize the memory slot

            
       
           
            if trainer.config.method=="pca":
                length=num_sample_per_task
            else:
                length=task_mem_size
            for ind in range(length):  # save it to the memory
                trainer.task_grad_memory[t].append(trainer.ogd_basis[:, ptr].cpu())
                ptr += 1
                
                
def orthonormalize(main_vectors, additional_vectors,device,normalize=True): 
    ## orthnormalize the basis (graham schmidt)
    for element in range(additional_vectors.size()[1]):
        
       
        coeff=torch.mv(main_vectors.t(),additional_vectors[:,element])  ## x - <x,y>y/ ||<x,y>||
        pv=torch.mv(main_vectors, coeff)
        d=(additional_vectors[:,element]-pv)/torch.norm(additional_vectors[:,element]-pv,p=2)
        main_vectors=torch.cat((main_vectors,d.view(-1,1)),dim=1)
        del pv
        del d
    return main_vectors.to(device)
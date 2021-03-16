import matplotlib as mpl
mpl.use('Agg')
import argparse
import numpy as np
import matplotlib.pyplot as plt
import torch
import random
import trainer
import pickle
import os
from utils.utils import get_benchmark_data_loader, test_error
from algos import ogd,ewc,gem
from tqdm.auto import tqdm



parser = argparse.ArgumentParser()

### Algo parameters
parser.add_argument("--seed", default=1, type=int)              # Sets Gym, PyTorch and Numpy seeds
parser.add_argument("--val_size", default=256, type=int)
parser.add_argument("--nepoch", default=10, type=int)             # Number of epoches
parser.add_argument("--batch_size", default=32, type=int)      # Batch size 
parser.add_argument("--memory_size", default=100, type=int)     # size of the memory
parser.add_argument("--hidden_dim", default=100, type=int)      # size of the hidden layer                 
parser.add_argument('--lr',default=1e-3, type=float)  
parser.add_argument('--n_tasks',default=15, type=int)  
parser.add_argument('--workers',default=2, type=int) 
parser.add_argument('--eval_freq',default=1000, type=int) 

## Methods parameters
parser.add_argument("--all_features",default=0, type=int) # Leave it to 0, this is for the case when using Lenet, projecting orthogonally only against the linear layers seems to work better

## Dataset
parser.add_argument('--dataset_root_path',default=" ", type=str,help="path to your dataset  ex: /home/usr/datasets/")
parser.add_argument('--subset_size',default=1000, type=int, help="number of samples per class, ex: for MNIST, \
            subset_size=1000 wil results in a dataset of total size 10,000") 
parser.add_argument('--dataset',default="split_cifar", type=str)
parser.add_argument("--is_split", action="store_true")
parser.add_argument('--first_split_size',default=5, type=int) 
parser.add_argument('--other_split_size',default=5, type=int)
parser.add_argument("--rand_split",default=False, action="store_true")
parser.add_argument('--force_out_dim', type=int, default=10,
                              help="Set 0 to let the task decide the required output dimension", required=False)

## Method
parser.add_argument('--method',default="ogd", type=str,help="sgd,ogd,pca,agem,gem-nt")

## PCA-OGD
parser.add_argument('--pca_sample',default=3000, type=int) 
## agem
parser.add_argument("--agem_mem_batch_size", default=256, type=int)     # size of the memory
parser.add_argument('--margin_gem',default=0.5, type=float)
## EWC
parser.add_argument('--ewc_reg',default=10, type=float) 
parser.add_argument('--fisher_sample',default=1024, type=int)

## Folder / Logging results
parser.add_argument('--save_name',default="result", type=str,  help="name of the file")

config = parser.parse_args()
config.device=torch.device("cuda" if torch.cuda.is_available() else "cpu")



np.set_printoptions(suppress=True)

config_dict=vars(config)

### setting seeds    
torch.manual_seed(config.seed)
np.random.seed(config.seed)
random.seed(config.seed)

torch.backends.cudnn.benchmark=True
torch.backends.cudnn.enabled=True

config.folder="method_{}_dataset_{}_memory_size_{}_bs_{}_lr_{}_epochs_per_task_{}".format(config.method, \
                                                config.dataset,config.memory_size,config.batch_size, config.lr,config.nepoch)
                                                                      


## create folder to log results
if not os.path.exists(config.folder):
    os.makedirs(config.folder, exist_ok=True)  
  
### name of the file
config.save_name=config.save_name+'_seed_'+str(config.seed)


### dataset path
# config.dataset_root_path="..."


########################################################################################
### dataset ############################################################################ 
print('loading dataset')
train_dataset_splits,val_loaders,task_output_space=get_benchmark_data_loader(config)
config.out_dim = {'All': config.force_out_dim} if config.force_out_dim > 0 else task_output_space


### loading trainer module
trainer=trainer.Trainer(config,val_loaders)





t=0
print('start training')
########################################################################################
### start training #####################################################################
for task_in in range(config.n_tasks):
    rr=0
   
    train_loader = torch.utils.data.DataLoader(train_dataset_splits[str(task_in+1)],
                                                       batch_size=config.batch_size,
                                                       shuffle=True,
                                                       num_workers=config.workers)
    ### train for EPOCH times
    print("================== TASK {} / {} =================".format(task_in+1, config.n_tasks))
    for epoch in tqdm(range( config.nepoch), desc="Train task"):
        
        
         trainer.ogd_basis.to(trainer.config.device) 
         
         for i, (input, target, task) in enumerate(train_loader):
          
            trainer.task_id = int(task[0])
            t+=1
            rr+=1
            inputs = input.to(trainer.config.device)
            target = target.long().to(trainer.config.device)
           
            out = trainer.forward(inputs,task).to(trainer.config.device)
            loss = trainer.criterion(out, target)
            
            if config.method=="ewc" and (task_in+1)>1:
                loss+=config.ewc_reg*ewc.penalty(trainer)                
            
            loss.backward()
            trainer.optimizer_step()
            ### validation accuracy
           
            if rr%trainer.config.eval_freq==0: 
                for element in range(task_in+1):
                    trainer.acc[element]['test_acc'].append(test_error(trainer,element))
                    trainer.acc[element]['training_steps'].append(t)    
    
        
    for element in range(task_in+1):
        trainer.acc[element]['test_acc'].append(test_error(trainer,element))
        trainer.acc[element]['training_steps'].append(t)
        print("  task {} / accuracy: {}  ".format(element+1, trainer.acc[element]['test_acc'][-1]))
        
    
    ## update memory at the end of each tasks depending on the method
    if config.method in ['ogd','pca']:
        trainer.ogd_basis.to(trainer.config.device)
        ogd.update_mem(trainer,train_loader,task_in+1)
        
    if config.method=="agem":
        gem.update_agem_memory(trainer,train_loader,task_in+1)
    
    if config.method=="gem-nt":  ## GEM-NT
        gem.update_gem_no_transfer_memory(trainer,train_loader,task_in+1)
    
    if config.method=="ewc":
        ewc.update_means(trainer)
        ewc.consolidate(trainer,ewc._diag_fisher(trainer,train_loader))
        
        
        

### Plotting accuracies
print('plotting accuracies')
plt.close('all')
for tasks_id in range(len(trainer.acc.items())):
    plt.plot(trainer.acc[tasks_id]['training_steps'],trainer.acc[tasks_id]['test_acc'])
plt.grid()
plt.savefig(config.folder+'/'+config.save_name+".png",dpi=72)




print('Saving results')
output = open(config.folder+'/'+config.save_name+'.p', 'wb')
pickle.dump(trainer.acc, output)
output.close()






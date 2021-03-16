import dataloaders.base
from dataloaders.datasetGen import SplitGen, PermutedGen,RotatedGen
from torch.nn.utils.convert_parameters import _check_param_device, parameters_to_vector, vector_to_parameters
import torch


        
def get_benchmark_data_loader(config):
    ## example: config.dataset_root_path='/home/usr/dataset/CIFAR100'
    config.dataset_root_path=''
    if config.dataset=="permuted":
        config.force_out_dim=10
        train_dataset, val_dataset = dataloaders.base.__dict__['MNIST'](config.dataset_root_path, False ,subset_size=config.subset_size)
        
        train_dataset_splits, val_dataset_splits, task_output_space = PermutedGen(train_dataset, val_dataset,config.n_tasks,remap_class= False)
        
    elif config.dataset=="rotated":
        config.force_out_dim=10
        import dataloaders.base
        
        Dataset = dataloaders.base.__dict__["MNIST"]
        n_rotate=config.n_tasks  
       
        rotate_step=5
      
      
        train_dataset_splits, val_dataset_splits, task_output_space = RotatedGen(Dataset=Dataset,
                                                                                          dataroot=config.dataset_root_path,
                                                                                          train_aug=False,
                                                                                          n_rotate=n_rotate,
                                                                                          rotate_step=rotate_step,
                                                                                          remap_class=False
                                                                                          ,subset_size=config.subset_size)
       
    elif config.dataset=="split_mnist":
        config.first_split_size=2
        config.other_split_size=2
        config.force_out_dim=0
        config.is_split=True
        import dataloaders.base
        Dataset = dataloaders.base.__dict__["MNIST"]
       
            
        
        if config.subset_size<50000:
            train_dataset, val_dataset = Dataset(config.dataset_root_path,False, angle=0,noise=None,subset_size=config.subset_size)
        else:
            train_dataset, val_dataset = Dataset(config.dataset_root_path,False, angle=0,noise=None)
        train_dataset_splits, val_dataset_splits, task_output_space = SplitGen(train_dataset, val_dataset,
                                                                               first_split_sz=config.first_split_size,
                                                                               other_split_sz=config.other_split_size,
                                                                               rand_split=config.rand_split,
                                                                               remap_class=True)
     
        config.n_tasks = len(task_output_space.items())


    elif config.dataset=="split_cifar":
        config.force_out_dim=0
        config.first_split_size=5
        config.other_split_size=5
        config.is_split=True
        import dataloaders.base
        Dataset = dataloaders.base.__dict__["CIFAR100"]
        # assert config.model_type == "lenet"  # CIFAR100 is trained with lenet only
    
        train_dataset, val_dataset = Dataset(config.dataset_root_path,False, angle=0)
        
        train_dataset_splits, val_dataset_splits, task_output_space = SplitGen(train_dataset, val_dataset,
                                                                                   first_split_sz=config.first_split_size,
                                                                                   other_split_sz=config.other_split_size,
                                                                                   rand_split=config.rand_split,
                                                                                   remap_class=True)
        config.n_tasks=len(train_dataset_splits)
    
    config.out_dim = {'All': config.force_out_dim} if config.force_out_dim > 0 else task_output_space
    
    val_loaders = [torch.utils.data.DataLoader(val_dataset_splits[str(task_id)],
                                                       batch_size=256,shuffle=False,
                                                       num_workers=config.workers)
                   for task_id in range(1, config.n_tasks + 1)]
    
    return train_dataset_splits,val_loaders,task_output_space



def test_error(trainer,task_idx):
        trainer.model.eval()
        acc = 0
        acc_cnt = 0
        with torch.no_grad():
            for idx, data in enumerate(trainer.val_loaders[task_idx]):
               
                    data, target, task = data
                   
                    data = data.to(trainer.config.device)
                    target = target.to(trainer.config.device)
        
                    outputs = trainer.forward(data,task)
        
                    acc += accuracy(outputs, target)
                    acc_cnt += float(target.shape[0])
        return acc/acc_cnt
    
    
def accuracy(outputs,target):
        topk=(1,)       
        with torch.no_grad():
                maxk = max(topk)
              
                _, pred = outputs.topk(maxk, 1, True, True)
                pred = pred.t()
                correct = pred.eq(target.view(1, -1).expand_as(pred))
             
                res = []
                for k in topk:
                    correct_k = correct[:k].view(-1).float().sum().item()
                    res.append(correct_k)
        
                if len(res)==1:
                    return res[0]
                else:
                    return res
                
                
                
def parameters_to_grad_vector(parameters):
    # Flag for the device where the parameter is located
    param_device = None
    vec = []
    for param in parameters:
        # Ensure the parameters are located in the same device
        param_device = _check_param_device(param, param_device)
        vec.append(param.grad.view(-1))
        
    return torch.cat(vec)

def count_parameter(model):
    return sum(p.numel() for p in model.parameters())



def grad_vector_to_parameters(vec, parameters):
    # Ensure vec of type Tensor
    if not isinstance(vec, torch.Tensor):
        raise TypeError('expected torch.Tensor, but got: {}'
                        .format(torch.typename(vec)))
    # Flag for the device where the parameter is located
    param_device = None
    # Pointer for slicing the vector for each parameter
    pointer = 0
    for param in parameters:
        # Ensure the parameters are located in the same device
        param_device = _check_param_device(param, param_device)
        # The length of the parameter
        num_param = param.numel()
        # Slice the vector, reshape it, and replace the old data of the parameter
        # param.data = vec[pointer:pointer + num_param].view_as(param).data
        param.grad = vec[pointer:pointer + num_param].view_as(param).clone()
        # Increment the pointer
        pointer += num_param  

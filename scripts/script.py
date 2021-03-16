import itertools
import  os


prefix_list = ["python main.py "]

def generate_command(params_dict,prefix_list):
    
    final_command=[]
    for key,value in params_dict.items():
        if not isinstance(value,list):
            params_dict[key]=[value]
    
    
    keys, values = zip(*params_dict.items())
    
    params_combination=[(keys,v) for v in itertools.product(*values)]
      
    for i in params_combination:
        
        command=prefix_list[0]+" "
        for element in range(len(i[0])):
            command+=" --"+str(i[0][element])+" "+str(i[1][element])
    
        final_command.append(command)
    
    return final_command
        
    
def replicate_results_split_cifar():
    params_dict={
        "nepoch":10,
        "memory_size":100,
        "batch_size":32,
        "lr":1e-3,
        "subset_size":1000,
        "memory_size":100,
        "n_tasks":20,
        "eval_freq":1000,
        "agem_mem_batch_size":256,
        "pca_sample":3000,
        "dataset":"split_cifar",
         "fisher_sample":1024,
        "hidden_dim":  200,
        "ewc_reg":25,
        "seed": [0,1,2,3,4],
        "method":["pca","ogd","ewc","sgd","agem"]
        }
    return params_dict


### rotated and permuted mnist
def replicate_results_rotated_mnist():
    params_dict={
        "nepoch":10,
        "memory_size":100,
        "batch_size":32,
        "lr":1e-3,
        "subset_size":1000,
        "memory_size":100,
        "n_tasks":15,
        "eval_freq":1000,
        "agem_mem_batch_size":256,
        "pca_sample":3000,
        "dataset":"split_cifar",
        "fisher_sample":1024,
        "hidden_dim":  100,
        "ewc_reg":10,
        "seed": [0,1,2,3,4],
        "method":["pca","ogd","ewc","sgd","agem"]
        }
    return params_dict


def replicate_results_split_mnist():
    params_dict={
        "nepoch":5,
        "memory_size":100,
        "batch_size":32,
        "lr":1e-3,
        "subset_size":2000,
        "memory_size":100,
        "n_tasks":5,
        "eval_freq":1000,
        "agem_mem_batch_size":256,
        "pca_sample":3000,
        "dataset":"split_cifar",
        "fisher_sample":1024,
        "hidden_dim":  100,
        "ewc_reg":10,
        "seed": [0,1,2,3,4],
        "method":["pca","ogd","ewc","sgd","agem"]
        }
    return params_dict


if __name__ == '__main__':
    
    
    command_split_cifar=generate_command(replicate_results_split_cifar(),prefix_list)
    command_split_mnist=generate_command(replicate_results_split_mnist(),prefix_list)
    command_rotated_mnist=generate_command(replicate_results_rotated_mnist(),prefix_list)
    
    carrier=[command_split_cifar,command_split_mnist,command_rotated_mnist]
    name=["command_cifar","command_split_mnist","command_rotated_mnist"]
    if not os.path.exists("commands"):
            os.makedirs("commands", exist_ok=True) 
    for it in range(len(carrier)):
         
        with open("commands/{}.sh".format(name[it]), "w") as outfile:
            outfile.write("\n".join(carrier[it])+"\n")
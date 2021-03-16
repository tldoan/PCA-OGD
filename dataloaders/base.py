import torchvision
from torchvision import transforms
from .wrapper import CacheClassLabel
import torchvision.transforms.functional as TF
import torch

class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean
        
    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean
    
    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)
    
    

class RotationTransform:
    """Rotate by one of the given angles."""

    def __init__(self, angle):
        self.angle = angle

    def __call__(self, x):
        return TF.rotate(x, self.angle)


def MNIST(dataroot, train_aug=False, angle=0,noise=None,subset_size=1000):
   
    normalize = transforms.Normalize(mean=(0.1000,), std=(0.2752,))  # for 32x32
    rotate=torchvision.transforms.RandomRotation((angle,angle),fill=(0,))

        
    val_transform = transforms.Compose([
            transforms.Pad(2, fill=0, padding_mode='constant'),
            rotate,
            transforms.ToTensor(),
            normalize,
        ])
        
   
    if train_aug:
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            rotate,
            transforms.ToTensor(),
            normalize,
        ])
   
    train_dataset = torchvision.datasets.MNIST(
        root=dataroot,
        train=True,
        download=True,
        transform=val_transform
    )
    
  
    from collections import defaultdict
   
    if subset_size<50000:
    ### will take subset_size element per class, so if set to 1,000 for MNIST the whole dataset size will be 1,000x10 = 10,000
        subset_size =subset_size
        cnt = defaultdict(lambda : subset_size)
        # print(cnt)
        new_data = list()
        new_targets = list()
        for x, label in train_dataset:
            if cnt[label] > 0 :
                new_data.append(x)
                new_targets.append(label)
                cnt[label] -= 1
        
                
        train_dataset.dataset = torch.stack(new_data)
        train_dataset.labels = torch.Tensor(new_targets)
        train_dataset.data = torch.stack(new_data)
        train_dataset.targets = torch.Tensor(new_targets)
        

        ds = torch.utils.data.TensorDataset(train_dataset.dataset, train_dataset.labels)
        ds.root=dataroot
        train_dataset = CacheClassLabel(ds)
    else:
        train_dataset = CacheClassLabel(train_dataset)
    
    val_dataset = torchvision.datasets.MNIST(
        dataroot,
        train=False,
        transform=val_transform
    )
    val_dataset = CacheClassLabel(val_dataset)
   
    return train_dataset, val_dataset


def CIFAR10(dataroot, train_aug=False, angle=0):
    normalize = transforms.Normalize(mean=[0.491, 0.482, 0.447], std=[0.247, 0.243, 0.262])
    rotate = RotationTransform(angle=angle)

    val_transform = transforms.Compose([
        rotate,
        transforms.ToTensor(),
        normalize,
    ])
    train_transform = val_transform
    if train_aug:
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            rotate,
            transforms.ToTensor(),
            normalize,
        ])

    train_dataset = torchvision.datasets.CIFAR10(
        root=dataroot,
        train=True,
        download=True,
        transform=train_transform
        )
    train_dataset = CacheClassLabel(train_dataset)

    val_dataset = torchvision.datasets.CIFAR10(
        root=dataroot,
        train=False,
        download=True,
        transform=val_transform
    )
    val_dataset = CacheClassLabel(val_dataset)

    return train_dataset, val_dataset


def CIFAR100(dataroot, train_aug=False, angle=0):
    normalize = transforms.Normalize(mean=[0.507, 0.487, 0.441], std=[0.267, 0.256, 0.276])

    val_transform = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])
    train_transform = val_transform
    if train_aug:
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])

    train_dataset = torchvision.datasets.CIFAR100(
        root=dataroot,
        train=True,
        download=True,
        transform=train_transform
    )
    train_dataset = CacheClassLabel(train_dataset)

    val_dataset = torchvision.datasets.CIFAR100(
        root=dataroot,
        train=False,
        download=True,
        transform=val_transform
    )
    val_dataset = CacheClassLabel(val_dataset)

    return train_dataset, val_dataset

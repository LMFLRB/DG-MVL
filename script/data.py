import torch, os
import numpy as np
from torch.utils.data import Dataset, Subset, DataLoader
from scipy.io import loadmat
from sklearn import preprocessing
from random import shuffle
from collections import Counter
from typing import Union, List

from .syn_data import syn_multiview_dataset

valid_datasets=['Caltech101-7', 
                'Caltech101-20', 
                'MSRC', 
                'Outdoor-Scene', 
                'XRMB', 
                'NUS_WIDE_SCENE', 
                "NUS_WIDE_OBJECT",
                'BCI_IV',  
                'handwritten',
                'LandUse-21',
                'bbc_sports',
                'Scene-15', 
                'cub_googlenet_doc2vec_c10',
                'yaleB_mtv',
                'animal',
                'PIE_face_10',
                'ORL_mtv',
                'TCGA',
                'ADNI_379']

class multiview_dataset(Dataset):
    enable_weights={dataset: True if dataset in 
                    ['NUS-WIDE-SCENE','NUS_WIDE_OBJECT'] 
                    else False for dataset in valid_datasets}
    def __init__(self, 
                 root_dir:str="G:\Data\Multi_view",
                 name:str="NUS_WIDE_OBJECT",
                 **kwargs):      
        if name not in valid_datasets:
            ValueError("please input valid name for dataset")
        self.name = name  
        self.enable_sample_weight = self.enable_weights[name]
        data = loadmat(os.path.join(root_dir, f"{name}.mat"))
        self.data = []
        self.feature_dims = []
        for i in range(len(data['X'][0])):
            transform = preprocessing.MinMaxScaler() #preprocessing.scale()
            normalized = transform.fit_transform(data['X'][0][i].astype(np.float32))
            # normalized = preprocessing.scale(data['X'][0][i].astype(np.float32))
            self.data.append(normalized)
            self.feature_dims.append(normalized.shape[-1])
        self.n_view = len(self.data)
        self.labels = np.squeeze(data['Y']) if np.min(data['Y']) == 0 else np.squeeze(data['Y']-1)
        self.weights, self.counts = self.make_weights_for_balanced_classes()
        # self.n_class= len(list(set(self.labels)))
        self.n_class= self.labels.max()+1
        if self.enable_sample_weight:
            counts = [self.counts[label] for label in range(self.n_class)]
            max_,min_=max(counts),min(counts)
            self.info=f"dataset {name} is class unbalanced with samples {max_}/{min_} for largest/smallest classes."
            print(self.info)
       
    def __getitem__(self, idx: int) -> torch.Tensor:
        views = [torch.from_numpy(view[idx]) for view in self.data]
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        weight= torch.tensor(self.weights[idx], dtype=torch.float)
        # sample = {'views': views, 'label': label}
        # return sample
        return (views, label, weight)
    
    def __len__(self) -> int:
        return len(self.labels)
     
    def make_weights_for_balanced_classes(self):
        def clip(data, n_classes):
            min_, max_ = 0.5/n_classes, 0.5*(n_classes-1)/n_classes
            if data<min_:
                return min_
            elif data>max_:
                return max_
            else:
                return data
        counts = Counter()    
        if self.enable_sample_weight:        
            for y in self.labels:
                y = int(y)
                counts[y] += 1

            n_classes = len(counts)
            weight_per_class = {str(y): 1./counts[y] for y in counts}
            total=sum(list(weight_per_class.values()))
            weight_per_class = {y: val/total for y, val in weight_per_class.items()}

            weights = torch.zeros(len(self.labels))
            for i, y in enumerate(self.labels):
                weights[i] = clip(weight_per_class[str(y)],n_classes)
        else:
            n_classes = len(list(set(self.labels)))
            weights = [1./n_classes]*len(self.labels)
            weight_per_class={str(i): 1./n_classes for i in range(n_classes)}
        
        self.weight_per_class = weight_per_class

        return np.array(weights), counts
 
class cache_dataset_into_cuda(Dataset):
    def __init__(self, dataset) -> None:
        super().__init__()
        for key, val in dataset.__dict__.items():
            setattr(self, key, val)
        self.data = dataset
        self._cache = dict()

    def __getitem__(self, index: int) -> torch.Tensor:
        if index not in self._cache:
            self._cache[index] = list(self.data[index])
            self._cache[index][0] = [view.cuda(non_blocking=True) for view in self._cache[index][0]]
            self._cache[index][1] = self._cache[index][1].cuda(non_blocking=True)
            if len(self._cache[index])>2 and self._cache[index][2] is not None:
                self._cache[index][2] = self._cache[index][2].cuda(non_blocking=True)
        return self._cache[index]

    def __len__(self) -> int:
        return len(self.data)

class LightData(Dataset):
    enable_weights={dataset: True if dataset in 
                    ['NUS-WIDE-SCENE','NUS_WIDE_OBJECT'] 
                    else False for dataset in valid_datasets}
    def __init__(
            self,
            data_params:dict={},
            num_workers: int = 0,
            pin_memory: bool = False,
            shuffle: bool = True,
            train_batch_size: int =256,
            val_batch_size: int =256,
            drop_last: bool=False,
            seed: int=None,
            cross_validation: bool=False,
            **kwargs
        ):
        super().__init__()

        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.drop_last = drop_last
        self.shuffle = shuffle
        self.cuda = data_params.cuda
        
        self.dataset = (syn_multiview_dataset(**data_params) if data_params.name == 'synthetic' 
                        else multiview_dataset(**data_params))
        if self.cuda:
            self.dataset = cache_dataset_into_cuda(self.dataset)
        self.__dict__.update(self.dataset.__dict__)
        
        self.length = len(self.dataset)

        self.seed = seed if seed is not None else 1266
        if cross_validation:
            self.n_cv = data_params.get('n_cross_validation')
            self.n_per_cv = int(self.length/self.n_cv)
            self.all_indics = self.random_split()
            # self.all_indics =list(range(self.length))
            self.get_current_cv_dataset()
        else:
            self.split_with_seed()
            
        self.train_batch_size  = train_batch_size if train_batch_size<self.length_train else self.length_train
        self.val_batch_size = val_batch_size if train_batch_size<self.length_val else self.length_val

    def get_current_cv_dataset(self, cv_idx:int=0):
        self.cv_idx = cv_idx
        step = self.n_per_cv
        split_idx = cv_idx*step
        next_step_idx = split_idx + step
        valid_indics = self.all_indics[split_idx:next_step_idx]
        train_indics = self.all_indics[:split_idx]+self.all_indics[next_step_idx:]
        self.train_dataset = Subset(self.dataset, train_indics)
        self.val_dataset = Subset(self.dataset, valid_indics)        
        self.length_train = len(train_indics)
        self.length_val = len(valid_indics)
        return self        

    def split_with_seed(self, seed: int=None, split=[0.7,0.3]):        
        if seed is None:
            seed = self.seed 
        g = torch.Generator()
        g.manual_seed(seed)
        all_indics = list(torch.randperm(self.length, generator=g))
        split_index = int(self.length*max(split))
        train_indics = all_indics[:split_index]
        valid_indics = all_indics[split_index:]
        self.train_dataset = Subset(self.dataset, train_indics)
        self.val_dataset = Subset(self.dataset, valid_indics)
        self.length_train = len(train_indics)
        self.length_val = len(valid_indics)
        
        return self
    
    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.train_batch_size,
            num_workers=self.num_workers,
            drop_last = self.drop_last,
            pin_memory=self.pin_memory,
            shuffle=self.shuffle,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.val_batch_size,
            num_workers=self.num_workers,
            drop_last=self.drop_last,
            pin_memory=self.pin_memory,
            shuffle=False,
        )
    
    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=144,
            num_workers=self.num_workers,
            drop_last=self.drop_last,
            pin_memory=self.pin_memory,
            shuffle=False,
        )
     
    def all_dataloader(self) -> DataLoader:
        return DataLoader(
            self.dataset,
            batch_size=self.train_batch_size,
            num_workers=self.num_workers,
            drop_last=self.drop_last,
            pin_memory=self.pin_memory,
            shuffle=False,
        )
    
    def random_split(self, seed=None):
        if seed is None:
            seed = self.seed 
        g = torch.Generator()
        g.manual_seed(seed)
        indices = list(torch.randperm(self.length, generator=g))
        return indices
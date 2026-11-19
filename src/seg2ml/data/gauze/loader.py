import logging
from abc import ABC
from typing import Any, Callable, List, Optional, Tuple, Union, Dict
import coloredlogs
import numpy as np
import pandas as pd
import torch
import pytorch_lightning as pl
from torch.utils.data.dataloader import DataLoader
from torchvision import datasets, transforms
from seg2ml.data.gauze.dataset import GauzeDataset

class GauzeDataModule(pl.LightningDataModule):
    def __init__(
        self,
        batch_size: int,
        split_config:Dict = None,
        transforms = None,
        num_workers: int = 1,
        collate_function: Optional[Callable] = None,
        **kwargs
    ):  
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.collate_function = collate_function
        self.current_training_phase = "main"

        self.init_transforms(transforms)
        self.prepare_dataset_splits(split_config, **kwargs)

    def switch_training(self, phase):
        self.current_training_phase = phase

    def full_dataloader(self):
        return self.init_dataloader(dataset=self.full_dataset, split= "full") if self.full_dataset else None
    
    def train_dataloader(self):
        if self.current_training_phase == "main":
            return self.init_dataloader(dataset=self.train_dataset, split= "train") if self.train_dataset else None
        else: 
            return self.init_dataloader(dataset=self.warmup_train_dataset, split= "train") if self.warmup_train_dataset else None
        
    def val_dataloader(self):
        return self.init_dataloader(dataset=self.valid_dataset, split="valid") if self.valid_dataset else None

    def test_dataloader(self):
        return self.init_dataloader(dataset=self.test_dataset, split="test") if self.test_dataset else None

    def test2_dataloader(self):
        return self.init_dataloader(dataset = self.test2_dataset, split="test2") if self.test2_dataset else None
    
    def init_transforms(self,transforms):
        if transforms:
            self.train_transforms = transforms.get("train", None)
            self.test_transforms = transforms.get("test", None)
            if not (self.train_transforms and self.test_transforms):
                print("assiging same pre-processing transformations to train and evaluation splits")
                self.train_transforms = self.test_transforms = transforms
            self.warmup_train_transforms = transforms.get("warmup_train", None)
        else: print("No pre-processing transformations provided")

    def prepare_dataset_splits(self, split_config, **kwargs):
        if split_config is None:
            print("Retrieving full dataset!!")
            self.full_dataset = self.init_dataset(
                split="full",
                patient_idxs = [],
                transforms = self.train_transforms, #in this case, train and transforms are same
                **kwargs)
            
        else:
            self.train_dataset = self.init_dataset(
            split="train",
            patient_idxs = split_config["train_idx"],
            transforms = self.train_transforms,
            **kwargs)
        
            if self.warmup_train_transforms:
                self.current_training_phase = "warmup"
                self.warmup_train_dataset = self.init_dataset(
                    split="train",
                    patient_idxs = split_config["train_idx"],
                    transforms = self.warmup_train_transforms,
                    **kwargs)
            
            self.valid_dataset = self.init_dataset(
                split="val",
                patient_idxs = split_config["val_idx"],
                transforms = self.test_transforms,
                **kwargs)
            
            self.test_dataset =self.init_dataset(
                split="test",
                patient_idxs = split_config["test_idx"],
                transforms = self.test_transforms,
                **kwargs)
            
            self.test2_dataset =self.init_dataset(
            
            
        
                split="test2",
                patient_idxs = split_config["test2_idx"],
                transforms = self.test_transforms,
                **kwargs)
            
    
    def init_dataloader(self, dataset, split = None):
        shuffle = True if split == "train" else False
        if dataset is not None:
            data_loader = DataLoader(
                dataset= dataset,
                batch_size=self.batch_size,
                collate_fn=self.collate_function,
                shuffle=shuffle,
                num_workers=self.num_workers,
                drop_last=False
            )
        print("{} samples in split {}".format(len(dataset), split))
        return data_loader
    
    def init_dataset(self, split, patient_idxs, *args, **kwargs):
        return GauzeDataset(split=split,  
                                patient_idxs=patient_idxs,
                                *args, **kwargs)      
        # full_dataset = self.init_dataset(
        #     split="train",
        #     video_indices = split_config["train_idx"],
        #     skip_frame_count = skip_frame_count,
        #     unique_classes = self.unique_classes,
        #     interpolation_annotation = interpolation_annotation,
        #     image_dim = self.image_dim,
        #     out_seg_channel = self.out_seg_channel,
        #     max_size=max_size,
        #     **kwargs
        # )

        # def split_gauze_dataset(full_dataset):
        #     print("Split ration of train:val:test = 0.8:0.2")
        #     data_len = len(full_dataset)
        #     per_idx = torch.randperm(data_len).numpy()
        #     train_idx = per_idx[:int(0.8*data_len)]
        #     val_idx = per_idx[int(0.8*data_len):]

        #     train_dataset = torch.utils.data.Subset(full_dataset, train_idx)
        #     val_dataset = torch.utils.data.Subset(full_dataset, val_idx)
    
        #     return train_dataset, val_dataset
        
        # self.train_dataset, self.val_dataset = split_gauze_dataset(self.train_dataset)

        
def main():
    transforms = {"data_transforms": [{"name": "Resize", 
                                    "module": "torchvision.transforms.v2",
                                    "args": {"size": (540, 960)}
                                    }]  ,

                "anno_transforms": [{"name": "ReplaceClassMapping", 
                                    "module": "seg2ml.utils.data_transforms",
                                    "args": {"label_mapping": {1: 0,  2: 0, 3: 0, 4: 1, 5: 2, 6: 3}}, 
                                     }, 
                                    {
                                    "name": "MapAnnotationBinary", 
                                    "module": "seg2ml.utils.data_transforms",
                                    "args": {"out_seg_channels": 4}, 
                                    }]                 
                 }
    split_config={"train_idx": [18], "val_idx": [9], "test_idx": [18], "test2_idx": [1]}
    kwargs = {"transforms": transforms, "data_root_dir": "./data"}
    
    dataloader = GauzeDataLoader(batch_size=1, split_config=split_config,**kwargs)                     
    print("\nbatch size: {}".format(dataloader.batch_size))
    print("number of batches for training: {}". format(len(dataloader.train)))
    print("number of batches for validation: {}". format(len(dataloader.validate)))

    for idx, batch in enumerate(dataloader.train):
        (input, target) = batch      
        print()
        print("\ninput.shape: ", input.shape)
        print("target.shape:", target.shape)
        import matplotlib.pyplot as plt
        plt.imshow(input[0].permute(1,2,0))
        plt.savefig('output_image_plot.png', bbox_inches='tight', pad_inches=0)
        break    
        
    print("done")
    
if __name__ == "__main__":
    main()

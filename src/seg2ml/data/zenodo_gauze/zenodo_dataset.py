import os
import glob
from typing import Any, Dict, List, Optional, Tuple, Union
from unittest import skip
import random
import torch
import pandas as pd
import numpy as np
from itertools import chain
from torch.utils.data import Dataset
from PIL import Image
from seg2ml.utils.data_transforms import GauzeTransforms
from seg2ml.utils.helper import hash_string
random.seed(10)
      
class GauzeZenodo(Dataset):
    def __init__(self, 
                 data_root_dir = "./data/zenodo",
                 cls_csv_name = "seg_paths.csv",
                 **kwargs
                 ):
        self.data_root_dir = data_root_dir
        self.cls_csv_name = cls_csv_name
       
   
    def prepare_split_paths(self, patient_idxs: Union[int, List[int]] = [],
                                  split: str = "full",):
        """Function to provide the paths to the inputs, annotations (gold and silver) and their respective labels
        Args:
            split: dataset split, train/val/test   
            max_size: max nuber of samples to be returned from the split
            
        Returns:
            List of paths to inputs, annotations and label 
        """
        input_paths, anno_paths = [], []

        path_dataframe = pd.read_csv(os.path.join(self.data_root_dir, self.cls_csv_name))

        if split != "full":
            if  isinstance(patient_idxs, int):
                patient_idxs = [patient_idxs]
            for patient_id in patient_idxs:
                pat_df = path_dataframe[path_dataframe["patient_ID"] == "ID"+str(patient_id)]
                input_paths.append(pat_df["img_path"].to_list())
                anno_paths.append(pat_df["anno_path"].to_list())
            input_paths = list(chain.from_iterable(input_paths))
            anno_paths = list(chain.from_iterable(anno_paths))
        else:
            input_paths = path_dataframe["img_path"].to_list()
            anno_paths = path_dataframe["anno_path"].to_list()
            
        print("\nSplit: ", split)
        print("Total Input frames: ", len(input_paths))
        print("Total Annotation frames: ", len(anno_paths))
        return input_paths, anno_paths
    
    def return_trans_data_path(self, split, img_size= None, out_seg_channels = None):
        hash_str = os.path.join(str(img_size), str(out_seg_channels))
        hash_name = hash_string(hash_str)
        return os.path.join(self.data_root_dir, "tranformed_data/",  split, hash_name)

    def save_tranformed_data(self, data_paths):
        for index in range(len(self.input_paths)):
            print("saving input at index: ", index)
            img_path =  self.input_paths[index]
            anno_path = self.anno_paths[index]

            input = Image.open(img_path)
            target = Image.open(anno_path)
            input, target = self.transforms(input, target)
            torch.save(input, os.path.join(data_paths[0],"input_"+str(index)+".pt"))
            torch.save(target,os.path.join(data_paths[1],"target_"+str(index)+".pt"))
    
    def load_trans_data(self, trans_data_path: str, split = "train"): 
            trans_inputs_path = os.path.join(trans_data_path,"inputs")
            trans_targets_path = os.path.join(trans_data_path,"targets")
            trans_paths_comb = [trans_inputs_path, trans_targets_path]
            print("Transformed data path: ", trans_data_path)
            if not os.path.exists(trans_data_path):
                print("\nPreparing transformed data for split: ", split)
                os.makedirs(trans_inputs_path)
                os.makedirs(trans_targets_path)
                self.save_tranformed_data(trans_paths_comb)
            self.input_paths = list(glob.glob(trans_inputs_path+"/*"))
            self.anno_paths = list(glob.glob(trans_targets_path+"/*"))
            assert len(self.input_paths) > 0, "Tranformed data path do not contain inputs!!"
    
class GauzeZenodoDataset(GauzeZenodo):
    def __init__(self, 
                 transforms = None, 
                 patient_idxs: List[int] = [] ,
                 split = "train",
                 save_transformed_data = False,
                 *args, **kwargs):
    
        super().__init__(*args, **kwargs)
        self.input_paths, self.anno_paths = self.prepare_split_paths(split = split, patient_idxs = patient_idxs)
        self.transforms = GauzeTransforms(transforms) if transforms else lambda x:x
        self.process_data = lambda im_path, anno_path: self.transforms(Image.open(im_path), Image.open(anno_path))
        
       
        if save_transformed_data:
            self.process_data = lambda im_path, anno_path: (torch.load(im_path), torch.load(anno_path))
            #trans_data_path = self.return_trans_data_path(split, image_dims, out_seg_channels)
            trans_data_path = self.return_trans_data_path(split)
            self.load_trans_data(trans_data_path=trans_data_path, split=split)
            
        self.total_inputs = len(self.input_paths)
        
        
    def __getitem__(self, index) -> Tuple[List[torch.Tensor], List[Any]]:
        if index >= len(self):
            raise IndexError
    
        input, target =  self.process_data(self.input_paths[index], self.anno_paths[index])
        # input, target =  self.process_data(self.input_paths[index], self.anno_paths[index])
        # input, target = self.transform(input, target)
        return input, target

    def __len__(self):
        return self.total_inputs
   
if __name__ == "__main__":
#### -------- Anno transformations are to be specified in the last, after data transforms---------  "
    transforms = {"data_transforms": [{"name": "Resize", 
                                    "module": "torchvision.transforms.v2",
                                    "args": {"size": (512, 512)}
                                    }],

                "anno_transforms": [
                                     {"name": "DropChannels", 
                                    "module": "seg2ml.utils.data_transforms",
                                    "args": {}, 
                                     },
                                     {"name": "ReplaceClassMapping", 
                                    "module": "seg2ml.utils.data_transforms",
                                    "args": {"label_mapping": {255: 1}}, 
                                     },
                                    ]                 
                 }
    
    kwargs = {"transforms": transforms,  "save_transformed_data": False, "data_root_dir": "./data/zenodo",}

    dataset = GauzeZenodoDataset(
        split="full", 
        **kwargs)


    print("length of dataset ", len(dataset))
    input, target = dataset[0]
    print("Input shape: {}".format(input.shape))
    print("Target shape: {}".format(target.shape))
    print("taret min and max values", target.min(), target.max())
    print("input type", input.dtype)
    print("target type", target.dtype)
    
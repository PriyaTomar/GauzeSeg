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
      
class Gauze(Dataset):
    def __init__(self, 
                 data_root_dir = "./data",
                 dataset_name = "dV_Pankreas_pseudonymisiert_cut",
                 cls_csv_name = "gauze_samples.csv",
                #  input_dir_name = "img",
                #  anno_dir_name = "masks_machine",
                 **kwargs
                 ):
        self.data_root_dir = data_root_dir
        self.dataset_name = dataset_name
        self.cls_csv_name = cls_csv_name
        # self.input_dir_name = input_dir_name
        # self.anno_dir_name = anno_dir_name
   
    def prepare_split_paths(self, patient_idxs: Union[int, List[int]],
                                  split: str,  
                                  max_size: int = None):
        """Function to provide the paths to the inputs, annotations (gold and silver) and their respective labels
        Args:
            split: dataset split, train/val/test   
            max_size: max nuber of samples to be returned from the split
            
        Returns:
            List of paths to inputs, annotations and label 
        """
        input_paths, anno_paths = [], []

        # path_dataframe =  pd.DataFrame(columns = [self.input_dir_name, self.anno_dir_name])
        # path_dataframe[self.input_dir_name] =  sorted(glob.glob(os.path.join(self.data_root_dir, self.dataset_name, self.input_dir_name, "*")))
        # path_dataframe[self.anno_dir_name] =  sorted(glob.glob(os.path.join(self.data_root_dir, self.dataset_name, self.anno_dir_name, "*")))
        
        # patient_id = path_dataframe[self.input_dir_name].str.split(pat=".", expand=True, n=2)[0].str.rsplit("_", expand=True, n=1)[1]
        # path_dataframe.insert(loc=0, column="patient_ID", value=patient_id)
        path_dataframe = pd.read_csv(os.path.join(self.data_root_dir, self.dataset_name, self.cls_csv_name))
        if  isinstance(patient_idxs, int):
            patient_idxs = [patient_idxs]
        for patient_id in patient_idxs:
            pat_df = path_dataframe[path_dataframe["patient_ID"] == "ID"+str(patient_id)]
            input_paths.append(pat_df["img_path"].to_list())
            anno_paths.append(pat_df["anno_path"].to_list())
            # input_paths.append(pat_df[self.input_dir_name].to_list())
            # anno_paths.append(pat_df[self.anno_dir_name].to_list())
        input_paths = list(chain.from_iterable(input_paths))
        anno_paths = list(chain.from_iterable(anno_paths))

        # randomly shuffle samples
        # merged = list(zip(input_paths, anno_paths))
        # random.shuffle(merged)
        # input_paths, anno_paths = zip(*merged)
        # input_paths, anno_paths = list(input_paths), list(anno_paths)
                
        if max_size:
            input_paths = input_paths[:max_size]
            anno_paths = anno_paths[:max_size]
            
        print("\nSplit: ", split)
        print("Total Input frames: ", len(input_paths))
        print("Total Annotation frames: ", len(anno_paths))
        return input_paths, anno_paths
    
    def return_trans_data_path(self, split, img_size= None, out_seg_channels = None):
        hash_str = os.path.join(str(img_size), str(out_seg_channels))
        hash_name = hash_string(hash_str)
        return os.path.join(self.data_root_dir, "tranformed_data/", self.dataset_name, split, hash_name)

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
    
class GauzeDataset(Gauze):
    def __init__(self, 
                 patient_idxs: List, 
                 transforms, 
                 split = "train",
                 max_size: Optional[int] = None,
                 save_transformed_data = False,
                 *args, **kwargs):
    
        super().__init__(*args, **kwargs)
        self.input_paths, self.anno_paths = self.prepare_split_paths(split = split, patient_idxs = patient_idxs, max_size=max_size)
        self.transforms = GauzeTransforms(transforms) if transforms else lambda x:x
        self.process_data = lambda im_path, anno_path: self.transforms(Image.open(im_path), Image.open(anno_path))
        
       
        if save_transformed_data:
            self.process_data = lambda im_path, anno_path: (torch.load(im_path), torch.load(anno_path))
            #trans_data_path = self.return_trans_data_path(split, image_dims, out_seg_channels)
            trans_data_path = self.return_trans_data_path(split)
            self.load_trans_data(trans_data_path=trans_data_path, split=split)
            
        self.total_inputs = len(self.input_paths)
        
        if max_size:
            self.total_inputs = max_size
        
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
                                    "args": {"size": (540, 960)}
                                    }]  ,

                "anno_transforms": [{"name": "ReplaceClassMapping", 
                                    "module": "seg2ml.data.utils.data_transforms",
                                    "args": {"label_mapping": {1: 0,  2: 0, 3: 0, 4: 1, 5: 2, 6: 3}}, 
                                     }, 
                                    {
                                    "name": "MapAnnotationBinary", 
                                    "module": "seg2ml.data.utils.data_transforms",
                                    "args": {"out_seg_channels": 4}, 
                                    }]                 
                 }
    
    kwargs = {"transforms": transforms, "max_size": 10, "save_transformed_data": True, "data_root_dir": "./data",}

    dataset = GauzeDataset(
        split="train", 
        patient_idxs = [18],
        **kwargs)


    print("length of dataset ", len(dataset))
    input, target = dataset[0]
    print("Input shape: {}".format(input.shape))
    print("Target shape: {}".format(target.shape))
    print("input type", input.dtype)
    print("target type", target.dtype)
    
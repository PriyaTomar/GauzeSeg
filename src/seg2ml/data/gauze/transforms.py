from typing import Dict, Tuple, Union, List, Optional, Literal
import torch
from PIL import Image
from seg2ml.utils.helper import create_class_instance
from torchvision.transforms import v2,  InterpolationMode
from torchvision.ops import masks_to_boxes
import random
import torchvision.transforms.functional as F
import numpy as np
import math

class GauzeTransforms:
    def __init__(self, 
                 transforms: Dict):
        self.transforms = self._init_transforms(transforms)
        
    def __call__(self, img: Image, anno: Image):
        img = v2.PILToTensor()(img)
        anno = v2.PILToTensor()(anno)
        for key, transform in self.transforms.items():
            if key == "data_transforms":
                channels = img.size()[0] 
                stacked_input = torch.concat((img, anno), dim=0)
                trans_stacked_input = transform(stacked_input)
                img, anno = trans_stacked_input[:channels], trans_stacked_input[channels:]
            elif key == "input_transforms": 
                img = transform(img)
            elif key == "anno_transforms":
                anno = transform(anno)
        img = v2.ToDtype(dtype = torch.float32, scale=True)(img)
        return img, anno

    def _init_transforms(self, transforms: Dict):
        """
        Initializes the list of transformations for input and annotation masks.
              The common transforms are applied first and then the transforms specif to anno or inputs are applied

        Parameters
        ----------
        params: Dict of Input, data and anno transforms, each containing a list of successive transforms for respective types

        Returns
        -------
        List of transforms:
        {
        Input_transforms:[]
        data:transforms:[]
        anno_transforms:[]
        }
        """
        
        def return_composed_transform(key: str, key_transforms: List[Dict]):
            composed_trans = []
            for params in key_transforms:
                composed_rand_apply = []
                if params["name"] in ["RandomApply"]:# transforms are list of transforms
                    for sub_trans_params in params["args"]["transforms"]:
                        composed_rand_apply.append(create_class_instance(sub_trans_params["module"], sub_trans_params["name"], sub_trans_params["args"]))
                    composed_trans.append(v2.RandomApply(transforms=composed_rand_apply, p=params["args"]["p"]))
                else:
                    composed_trans.append(create_class_instance(params["module"], params["name"], params["args"]))
            return v2.Compose(composed_trans)

        def shuffle_transformation_order(d, key = "data_transforms"):
            if key in d:
                # Create a new dictionary with the specified key first
                new_dict = {key: d[key]}
                # Add the remaining items
                new_dict.update((k, d[k]) for k in d if k != key)
                return new_dict
            else:
                raise KeyError(f"Key '{key}' not found in the transformation dictionary")

        composed_transforms = {}
        # move data_transform to first place in the transforms
        shuffled_trans = shuffle_transformation_order(transforms)
        for key, val in shuffled_trans.items(): #eg data_transforms
            if val:
                composed_transforms.update({key: return_composed_transform(key=key, key_transforms=val)})
        return composed_transforms

    @property
    def config(self) -> Dict:
        return {"name": "GauzeTransforms", "out_seg_channel": self.out_seg_channels}

class ReplaceClassMapping:
    """transforms to replace a class label according to the priority list of labels 
    """
    def __init__(self, label_mapping: dict):
        self.label_mapping = label_mapping

    def __call__(self, img: torch.tensor):
        map_img = torch.zeros(size=img.size(), dtype=img.dtype)
        for prev, new in self.label_mapping.items():
            map_img[img == prev] = new
        assert map_img.max() <= max(list(self.label_mapping.values())), "mapped image contains extra labels than specified!!"
        return map_img
    
class MapAnnotationBinary:
    """ Map the annotation  to array of 0 and 1 with the shape (C,W,H) 
    Args:
        Transform (_type_): _description_
    """
    def __init__(self, out_seg_channels:int=1):
        self.out_seg_channels = out_seg_channels
        
    def __call__(self, frame: torch.tensor):
        _, h,w = frame.shape
        seg = torch.zeros((self.out_seg_channels, h, w))

        for i in range(0, self.out_seg_channels):
            seg[i][frame.squeeze() == i] = 1
        return seg.to(torch.float)
    
class CropBlackBorders:
    """
        Crops out the black borders from all sides of an image based on pixel intensity using PyTorch tensors.

        Args:
            image (torch.tensor): Image tensor.
            threshold (int): Pixel intensity threshold to consider as non-black (0-255).
                            Default is 10, treating very dark pixels as black.

        Returns:
            cropped_image (torch.tensor): Cropped image without black borders.
    """
    def __init__(self) -> None:
        self.gray_transform = v2.Grayscale(num_output_channels=1)
           
    def __call__(self, stacked_input, threshold=10): 
        #image contains stacked RGB image and mask
        channels, height, width = stacked_input.shape
        assert channels>3, "Number of channels are less than Color channels! "
        image, anno = stacked_input[:3], stacked_input[3:]
        gray_tensor = self.gray_transform(image).squeeze(0)
        # Threshold to find non-black pixels (greater than the threshold)
        non_black_columns = torch.where(torch.max(gray_tensor, dim=0)[0] > threshold / 255.0)[0]
        non_black_rows = torch.where(torch.max(gray_tensor, dim=1)[0] > threshold / 255.0)[0]

        # If no non-black pixels are found, return the original image
        if non_black_columns.size(0) == 0 or non_black_rows.size(0) == 0:
            return image

        # Get the crop boundaries
        left_bound = non_black_columns[0].item()
        right_bound = non_black_columns[-1].item()
        top_bound = non_black_rows[0].item()
        bottom_bound = non_black_rows[-1].item()
        return torch.concat((image[:, top_bound:bottom_bound+1, left_bound:right_bound+1], 
                            anno[:, top_bound:bottom_bound+1, left_bound:right_bound+1]),
                            dim=0)
    
class ResizeToSquare:
    """ function to pad the images and resize it to sqaure to keep the aspect ratio unchanged"""
    def __init__(self, resize_size):
        self.resize_trans = v2.Resize(size=resize_size, interpolation=InterpolationMode.NEAREST)

    def __call__(self, image: torch.tensor):
        # Get original dimensions
        _, height, width = image.shape
        
        # Determine the size for the new square image
        size = max(width, height)
        
        # Calculate padding
        pad_width = size - width
        pad_height = size - height
        
        # Apply padding (left, top, right, bottom)
        pad_img = F.pad(image, (0, pad_height, pad_width, 0), fill=0, padding_mode='constant')
        #pad_mask = F.pad(image, (0, pad_height, pad_width, 0), fill=0, padding_mode='constant')
        return self.resize_trans(pad_img)
    
class PadAndRandomCrop:
    """ transform to pad the images and crop it to provided size to keep the aspect ratio unchanged"""
    def __init__(self, crop_size):
        self.rand_crop_transform = v2.RandomCrop(crop_size)

    def __call__(self, image):
        _, height, width = image.shape
                
        # Determine the size for the new square image
        size = max(width, height)
        
        # Calculate padding
        pad_width = size - width
        pad_height = size - height
        
        # Apply padding (left, top, right, bottom)
        pad_img = F.pad(image, (0, pad_height, pad_width, 0), fill=0, padding_mode='constant')
        return self.rand_crop_transform(pad_img)
    
    # class CustomRandomCrop:
    #     """ Transformation to pad the image and crop by keeping the aspect ration unchanged"""
    # def __init__(self, size: List):
    #     self.crop_trans = v2.RandomCrop(size=size)
    #     self.h = size[0]
    #     self.w = size[1]

    # def __call__(self, image: torch.tensor):
    #     # Get original dimensions
    #     _, height, width = image.shape
        
    #     # Determine the difference in crop
    #     h_diff = math.copysign(self.h - height, 1)
    #     w_diff = math.copysign(self.w - width, 1)
    #     if h_diff == -1.0 or w_diff == -1.0:
    #         size = max(width, height)
    #         # Calculate padding
    #         pad_width = size - width
    #         pad_height = size - height
        
    #         # Apply padding (left, top, right, bottom)
    #         image= F.pad(image, (0, pad_height, pad_width, 0), fill=0, padding_mode='constant')
    #     return self.crop_trans(image)
    
class CropPadClassBox:
    """
    Transformation to extract bounding boxes of specific classes then crop and pad the bounding box to a sqaure size around the bounding box in the image. 
    """
    def __init__(self, 
                 class_labels: Union[int, List[int]], 
                 label_replace_value: Optional[int] = 0,
                 pad_strategy: Literal["box", "image"] = "image"):
        self.class_labels = class_labels
        self.label_replace_value = label_replace_value
        self.pad_strategy = pad_strategy

    def __call__(self,  stacked_input: torch.tensor):
        channels, img_h, img_w = stacked_input.shape
        assert channels>3, "Number of channels are less than Color channels! "
        img, mask = stacked_input[:3], stacked_input[3:]
        if self.class_labels:
            mask = self.extract_class_masks(mask, self.class_labels, self.label_replace_value)
        boxes = masks_to_boxes(mask).to(torch.int)
        #randomly select one of the masks
        random_idx = np.random.randint(0, len(boxes))
        box = boxes[random_idx]
        img_h, img_w = img.shape[1:]
        x_min, y_min, x_max, y_max = box
        #The boxes are in (x1, y1, x2, y2) format with 0 <= x1 < x2 and 0 <= y1 < y2.
        box_height = y_max - y_min 
        box_width = x_max - x_min

        if box_height > box_width:
            # pad width
            pad = box_height - box_width
            x_min, x_max, y_min, y_max = self.pad_bounding_box(pad=pad, x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max, img_w= img_w, pad_strategy=self.pad_strategy)

        elif box_height < box_width: 
            pad = box_width - box_height
            y_min, y_max, x_min, x_max = self.pad_bounding_box(pad=pad, x_min=y_min, x_max=y_max, y_min= x_min, y_max=x_max,  img_w=img_h, pad_strategy=self.pad_strategy)

        # Crop the image using the bounding box
        cropped_image_tensor = img[:, y_min:y_max, x_min:x_max]
        cropped_anno_tensor = mask[:, y_min:y_max, x_min:x_max]
        return torch.concat((cropped_image_tensor, cropped_anno_tensor),dim=0)
    
    def extract_class_masks(self, mask: torch.tensor, class_labels: Optional[List], label_replace_value: Optional[int] = 0):
        """
        Function to remove the labels not in class labels and replace them with replace value.
        Args:
            - mask: annotation mask
            - class_labels: labels to keep in mask
            - replace_value: value for the redundant/extra labels
        """

        mask_labels = mask.unique()
        extra_labels = [label for label in mask_labels if label not in class_labels]
        for label in extra_labels:
            mask[mask == label] = label_replace_value
        return mask
    
    def pad_bounding_box(self, pad, x_min, x_max, y_min, y_max, img_w, pad_strategy):
        """
        Function to pad the bounding box from the nearby pixels to return a sqaure box dimesnions. Given bopunding box is assumed to need padding in the horizontal direction.
            Case 1: if there is more space than padding in the image to the right side of bounding box, then randomly select the pad on left and remaining from right side.
            Case 2: same as case 1 but flexible padding is on left side
            Case 3: left and right space in the images is enough for padding then randomly select pad on any one side and remaining on other
            Case 4: Space is not enough for padding, so crop the bounding box area and match the heigth of box to the width(smaller dimension) of box

        Args: 
            - pad: padding needed around bounding box in direction of smaller dimension
            - x_min, x_max, y_min, y_max: dimesnions of the given bounding box
            - img_w: width of image (smaller dimension in which padding is to be done)
        """
        left_space = x_min
        right_space = abs(img_w - x_max)
        condition_left = left_space >= pad
        condition_right = right_space >= pad
        flip = np.random.randint(0, 2)
        box_width =  x_max - x_min if pad_strategy == "box" else img_w
        max_left_pad = min(pad, left_space) if pad_strategy == "box" else left_space
        max_right_pad = min(pad, right_space) if pad_strategy == "box" else right_space

        if condition_left and condition_right:
            condition_right = False if flip else True # Case1 or Case2
    
        #Case1:
        if condition_right: 
            #"pad right based on min of pad and left space, for only left_space, degree of movement will be larger"
            left_pad = np.random.randint(low = 0, high = max_left_pad+1)
            x_min -= left_pad
            x_max+= (pad -left_pad)
        #Case2: 
        elif condition_left:
            #pad left
            right_pad = np.random.randint(low = 0, high = max_right_pad+1)
            x_min = x_min - (pad - right_pad)
            x_max+= right_pad
        #Case3:
        elif (left_space + right_space) >=pad:
            #pad full left
            if not flip:
                x_min -= left_space
                x_max += pad-left_space 
            else: 
                #move full right
                x_max+= right_space
                x_min -= pad - right_space
        #Case4:
        else: 
            if not flip: #move max
                y_max = y_min + box_width
            else: y_min = y_max - box_width  #move min
        return x_min, x_max, y_min, y_max
    
class DropChannels:
    """
    Drop the channels mask in case the annotations mask has 3 equalchannels
    # like in the gauze segmentation dataset downloaded from zenodo
    """
    def __init__(self):
        pass
        
    def __call__(self, anno_input: torch.tensor):
        if anno_input.shape[0] == 3:
            return torch.unsqueeze(anno_input[0], dim=0)

        else:
            raise ValueError("Number of channels in annotation mask is not 3")

class CustomCrop:
    def __init__(self, top_crop:int, left_crop:int, bottom_crop:int, right_crop:int):
        """
        Crop an image from all four sides.

        Args:
            top_crop (int): Number of pixels to crop from the top.
            left_crop (int): Number of pixels to crop from the left.
            bottom_crop (int): Number of pixels to crop from the bottom.
            right_crop (int): Number of pixels to crop from the right.

        Returns:
            PIL Image: The cropped image.
        """
        
        # Calculate the crop box
        self.top_crop = top_crop
        self.left_crop = left_crop
        self.right_crop = right_crop
        self.bottom_crop = bottom_crop

    def __call__(self, image):
        _, height, width = image.shape

        # Ensure that the crop dimensions are within the image bounds
        self.top = max(self.top_crop, 0)
        self.left = max(self.left_crop, 0)
        self.right = min(width, width - self.right_crop)
        self.bottom = min(height, height - self.bottom_crop)
        
        if (height, width) == (576, 720):
            crop_img = image[:, self.top : self.bottom, : ]
            return crop_img
        return image[:, self.top : self.bottom, self.left : self.right]

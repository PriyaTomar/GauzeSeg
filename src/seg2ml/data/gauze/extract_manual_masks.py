""" The gauze dataset has some objects annotated manually and some are auto-tracked. 
This script will extract only the images and corresponding masks which are not autotracked, to avoid any noise in the annotations. """

import json
import os
import shutil

def has_autotracked_object(json_file_path):
    """
    Checks if a given JSON annotation file contains any auto-tracked objects.

    Args:
        json_file_path (str): Path to the JSON annotation file.

    Returns:
        bool: True if the file contains any auto-tracked objects, False otherwise.
    """
    with open(json_file_path, "r") as f:
        json_data = json.load(f)

    # Check if any object has the "auto-tracked" tag
    for obj in json_data.get("objects", []):
        for tag in obj.get("tags", []):
            if tag["name"] == "auto-tracked":
                return True  # Found an auto-tracked object
    return False  # No auto-tracked objects

def filter_images_without_autotracked(json_dir, images_dir, masks_machine_dir, masks_human_dir, output_dir):
    """
    Filters images and masks without auto-tracked objects and copies them to a new directory.

    Args:
        json_dir (str): Directory containing JSON annotation files.
        images_dir (str): Directory containing corresponding image files.
        masks_dir (str): Directory containing segmentation mask files.
        output_dir (str): Output directory to save filtered images and masks.
    """
    # Create directories for filtered images and masks
    filtered_images_dir = os.path.join(output_dir, "img")
    filtered_masks_dir = os.path.join(output_dir, "masks_machine")
    filtered_human_mask_dir = os.path.join(output_dir, "masks_human")
    filtered_ann_dir = os.path.join(output_dir, "ann")

    
    os.makedirs(filtered_images_dir, exist_ok=True)
    os.makedirs(filtered_masks_dir, exist_ok=True)
    os.makedirs(filtered_human_mask_dir, exist_ok=True)
    os.makedirs(filtered_ann_dir, exist_ok=True)

    # Iterate over all JSON files in the annotation directory
    file_count = 0
    for json_file in os.listdir(json_dir):
        if json_file.endswith(".json"):
            json_path = os.path.join(json_dir, json_file)
            
            # Check if this annotation has any auto-tracked objects
            if not has_autotracked_object(json_path):
                # If no auto-tracked objects, move corresponding image and mask
                #annotation mask file_name: 268633_311871_ID6.mp4_36700.jpg.json 
                base_filename = os.path.splitext(json_file)[0]
                
                image_file = base_filename  
                mask_file =  os.path.splitext(base_filename)[0] + ".png"  # human and machine mask has .png format and same file name
                
                image_path = os.path.join(images_dir, image_file)
                masks_machine_path = os.path.join(masks_machine_dir, mask_file)
                masks_human_path = os.path.join(masks_human_dir, mask_file)
                
                # Copy image and mask if they exist
                if os.path.exists(image_path) and os.path.exists(masks_human_dir):
                    shutil.copy(image_path, filtered_images_dir)
                    shutil.copy(masks_machine_path, filtered_masks_dir)
                    shutil.copy(masks_human_path, filtered_human_mask_dir)
                    shutil.copy(json_path, filtered_ann_dir)
                    print(f"Copied: {image_file} and {mask_file}")
                    file_count +=1
    print("Total copied files: ", file_count)

if __name__ == "__main__":

    json_annotations_dir = "./data/dV_Pankreas_pseudonymisiert_cut/ann/"
    images_directory = "./data/dV_Pankreas_pseudonymisiert_cut/img/"
    masks_directory = "./data/dV_Pankreas_pseudonymisiert_cut/masks_machine/"
    masks_human_directory = "./data/dV_Pankreas_pseudonymisiert_cut/masks_human/"
    output_directory = "./data/manual_annotations/"

    filter_images_without_autotracked(json_annotations_dir, images_directory, masks_directory, masks_human_directory, output_directory)

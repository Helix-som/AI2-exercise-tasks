"""
Exercise 05 - Detection Dataset
==================================
Custom Dataset for object detection using COCO JSON annotations.

Unlike classification where __getitem__ returns (image, label),
detection returns (image, target_dict) where target contains
bounding boxes, labels, areas, and image IDs for all objects.

Run this script to verify:
    python src/dataset.py
"""

import os
import json
from collections import defaultdict

import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader


class PhenoBenchDetectionDataset(Dataset):
    """
    Detection dataset for PhenoBench using COCO JSON annotations.

    Args:
        root_dir (str): Path to split folder (e.g. 'data/phenobench/train').
        annotation_file (str): Path to COCO JSON file.
        transforms (callable, optional): Detection transforms operating
            on (image, target) pairs.
    """

    def __init__(self, root_dir, annotation_file, transforms=None):
        self.root_dir = root_dir
        self.transforms = transforms

        
    
        #   Step 1: Open annotation_file in json format.
        #           The JSON contains three keys: "images", "annotations",
        #           "categories".
        with open(annotation_file, "r") as f:
            coco_data = None # TODO: Remove "None" and add your code; Hint: Use json.load()
        #
        #   Step 2: Store the image list as self.images and the category
        #           list as self.categories.
        self.images = None # TODO: Remove "None" and extract "images" from coco_data dictionary
        self.categories = None # TODO: Remove "None" and extract "categories" from coco_data dictionary
        #
        #   Step 3: Loop over all "annotations" from coco_data dictionary — each has an "image_id"
        #           field. Append each annotation to the list for its image_id. 
        #           This way, self.img_to_anns[some_id] gives
        #           you all objects in that image.
    
        self.img_to_anns = defaultdict(list)
        for ann in coco_data["annotations"]:
            self.img_to_anns[ann["image_id"]].append(ann)
        

    def __len__(self):
        # TODO: Return the number of images in the dataset.
        #       Hint: self.images is a list.
        pass

    def __getitem__(self, idx):
        """
        Load one image and its detection targets.

        Returns:
            image: Image tensor.
            target (dict): Contains boxes, labels, image_id, area, iscrowd.
        """
        # Step 1: Extract the image information
        img_info = None # TODO: Remove "None" and get the image info dict from self.images[idx]
        image_id = None # TODO: Remove "None" and extract "id" from img_info
        file_name = None # TODO: Remove "None" and extract "file_name" from img_info
    
        # Step 2: Build the full path to the image file and open the image.
        img_path = os.path.join(self.root_dir, "images", file_name)
        image = None # TODO: Remove "None" and open the image with PIL(Image.open()) and convert to "RGB"
        
        # Step 3: Get all annotations for this image using
        annotations = None #TODO: Remove "None" and extract the annotations from self.img_to_anns using "image_id"
        
        # Step 4: Loop over the annotations. For each one:
        #         - Extract the bbox field: [x, y, w, h] (COCO format)
        #         - Convert the bbox format from PascalVOC to COCO [x1, y1, x2, y2] --> x1=x, y1=y, x2=x+w, y2=y+h
        #         - Collect the category_id as the label
        #         - Collect the area value
        #         - Collect iscrowd (default to 0 if missing)
        #         - Append each to separate lists (boxes, labels, areas, iscrowd).
        boxes = []
        labels = []
        areas = []
        iscrowd = []
        for ann in annotations:
            x, y, w, h = ann["bbox"]
            boxes.append([x, y, x + w, y + h])
            labels.append(ann["category_id"])
            areas.append(ann["area"])
            iscrowd.append(ann.get("iscrowd", 0))
            
        
        # Step 5: Convert lists to torch tensors and build the target dict:
        #         - "boxes":    FloatTensor of shape [N, 4]
        #         - "labels":   Int64Tensor of shape [N]
        #         - "image_id": Int64Tensor of shape [1]
        #         - "area":     FloatTensor of shape [N]
        #         - "iscrowd":  UInt8Tensor of shape [N]
        #         If no annotations exist for this image, create empty
        #         tensors with the correct shapes (e.g. torch.zeros((0, 4))
        #           for boxes).
        if len(boxes) > 0:
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            labels = torch.as_tensor(labels, dtype=torch.int64)
            areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])  # areas = (x_max - x_min) * (y_max - y_min)
            keep = areas > 1.0  # drop boxes with area ≤ 1 pixel
            boxes = boxes[keep]
            labels = labels[keep]
            areas = areas[keep]
            iscrowd = torch.as_tensor(iscrowd, dtype=torch.uint8)[keep]
            target = {
            "boxes": boxes,           # already a tensor from filtering
            "labels": labels,         # already a tensor
            "image_id": torch.tensor([image_id], dtype=torch.int64),
            "area": areas,            # already a tensor
            "iscrowd": iscrowd,       # already a tensor # With iscrowd=0, the annotation is a single object, With iscrowd=1, the annotation covers a collection of objects
            }
        else:
            target = {
                "boxes": torch.zeros((0, 4), dtype=torch.float32),
                "labels": torch.zeros((0,), dtype=torch.int64),
                "image_id": torch.tensor([image_id], dtype=torch.int64),
                "area": torch.zeros((0,), dtype=torch.float32),
                "iscrowd": torch.zeros((0,), dtype=torch.uint8),
            }
        
        # Step 6: If self.transforms is not None, apply them:
        if self.transform is not None:
            image, target = None # # TODO: Remove "None" and apply transformations. Hint: self.transforms()          
        
        return (image, target)
         


# ===========================================================================
# Verification
# ===========================================================================
if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from transforms import get_train_transforms
    from utils import collate_fn

    data_root = "data/phenobench_256"

    dataset = PhenoBenchDetectionDataset(
        root_dir=os.path.join(data_root, "train"),
        annotation_file=os.path.join(data_root, "annotations", "train.json"),
        transforms=get_train_transforms(),
    )

    print(f"Total images: {len(dataset)}")
    print(f"Categories: {dataset.categories}")

    image, target = dataset[0]
    print(f"\nSample 0:")
    print(f"  Image shape: {image.shape}")
    print(f"  Number of objects: {len(target['boxes'])}")
    print(f"  Boxes shape: {target['boxes'].shape}")
    print(f"  Labels: {target['labels']}")
    print(f"  Image ID: {target['image_id']}")

    loader = DataLoader(dataset, batch_size=2, shuffle=True,
                        num_workers=0, collate_fn=collate_fn)
    images, targets = next(iter(loader))

    print(f"\nBatch:")
    print(f"  Number of images: {len(images)}")
    print(f"  Image 0 shape: {images[0].shape}")
    print(f"  Targets 0 objects: {len(targets[0]['boxes'])}")
    print(f"  Targets 1 objects: {len(targets[1]['boxes'])}")

    print("\n--- Dataset verified. ---")

"""
Exercise 05 - Detection Transforms
=====================================
Transforms for object detection must handle both the image AND the
bounding boxes. If an image is flipped, the boxes must be flipped too.

Run this script to verify:
    python src/transforms.py
"""

import torch
import torchvision.transforms.functional as F
import random


class Compose:
    """Chain multiple detection transforms together."""

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target


class ToTensor:
    """Convert PIL image to tensor. Boxes remain unchanged."""

    def __call__(self, image, target):
        image = F.to_tensor(image)
        return image, target


class Normalize:
    """Normalize image tensor. Boxes remain unchanged."""

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, image, target):
        image = F.normalize(image, mean=self.mean, std=self.std)
        return image, target


class RandomHorizontalFlip:
    """
    Randomly flip image and boxes horizontally with given probability.
    Boxes are in [x1, y1, x2, y2] format.
    """

    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, image, target):
        #   Step 1: Generate a random number (random.random()) and only
        #           proceed if it falls below self.prob.
        rand_num = random.random()
        
        #   Step 2: Flip the image using F.hflip().
        if rand_num < self.prob:
            image = F.hflip(image)
        
            #   Step 3: Get the image width. Note that the image may be a
            #           PIL image or a tensor at this point depending on
            #           transform ordering — check which type it is and
            #           use .width (PIL) or .shape[-1] (tensor) accordingly.
            if isinstance(image, torch.Tensor):
                    width = image.shape[-1]
                else:
                    width = image.width
                
            #   Step 4: Mirror the bounding boxes.
            boxes = target["boxes"]
            new_boxes = boxes.clone()
            new_boxes[:, 0] = width - boxes[:, 2]
            new_boxes[:, 2] = width - boxes[:, 0]
        
            #   Step 5: Update target["boxes"] with the flipped boxes.
            target["boxes"] = new_boxes

        return image, target


class Resize:
    """
    Resize image so the shorter side equals min_size.
    Scale all bounding box coordinates proportionally.
    """

    def __init__(self, min_size, max_size=None):
        self.min_size = min_size
        self.max_size = max_size or min_size

    def __call__(self, image, target):
        #   Step 1: Get the original height and width. Same type check
        #           as in RandomHorizontalFlip — PIL uses .size (returns
        #           w, h), tensor uses .shape[-2:] (returns h, w).
        if isinstance(image, torch.Tensor):
            h, w = image.shape[-2:]
        else:
            w, h = image.size
        #
        #   Step 2: Compute the scale factor so the shorter side becomes
        #           self.min_size. If self.max_size is set, ensure the
        #           longer side doesn't exceed it.
        scale = self.min_size / min(h, w)
        if self.max_size is not None:
            scale = min(scale, self.max_size / max(h, w))
        #
        #   Step 3: Compute new_h and new_w by multiplying originals
        #           by the scale factor. Resize the image using
        #           F.resize(image, [new_h, new_w]).
        new_h = int(h * scale)
        new_w = int(w * scale)
        image = F.resize(image, [new_h, new_w])
        #   Step 4: Compute separate scale factors for x and y:
        #               scale_x = new_w / old_w
        #               scale_y = new_h / old_h
        #           Multiply all box x-coordinates (columns 0 and 2)
        #           by scale_x, and y-coordinates (columns 1 and 3)
        #           by scale_y.
        scale_x = new_w / w
        scale_y = new_h / h
        boxes = target["boxes"]
        boxes[:, [0, 2]] *= scale_x
        boxes[:, [1, 3]] *= scale_y
        target["boxes"] = boxes
        #
        #   Step 5: Recompute target["area"] from the scaled boxes:
        #           area = (x2 - x1) * (y2 - y1).
        target["area"] = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])

        return image, target


def get_train_transforms():
    """Training transforms with augmentation."""
    return Compose([
        ToTensor(),
        RandomHorizontalFlip(prob=0.5),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def get_val_transforms():
    """Validation transforms — deterministic, no augmentation."""
    return Compose([
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


# ===========================================================================
# Verification
# ===========================================================================
if __name__ == "__main__":
    from PIL import Image

    dummy_img = Image.new("RGB", (256, 256), color=(100, 150, 200))
    dummy_target = {
        "boxes": torch.tensor([[100, 200, 300, 400], [500, 600, 700, 800]], dtype=torch.float32),
        "labels": torch.tensor([1, 2], dtype=torch.int64),
        "area": torch.tensor([200 * 200, 200 * 200], dtype=torch.float32),
        "iscrowd": torch.tensor([0, 0], dtype=torch.uint8),
        "image_id": torch.tensor([1]),
    }

    print("Original boxes:", dummy_target["boxes"])

    train_tf = get_train_transforms()
    img_out, tgt_out = train_tf(dummy_img, dummy_target.copy())
    print(f"After train transforms — image shape: {img_out.shape}, boxes: {tgt_out['boxes']}")

    val_tf = get_val_transforms()
    img_out, tgt_out = val_tf(dummy_img, dummy_target.copy())
    print(f"After val transforms — image shape: {img_out.shape}, boxes: {tgt_out['boxes']}")

    print("\n--- Transforms verified. ---")

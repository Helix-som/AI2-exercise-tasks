"""
Exercise 05 - Detection Training Script
==========================================
Training loop for object detection. Key differences from classification:
  - Model computes loss internally (returns loss_dict in train mode).
  - No explicit criterion needed.
  - Validation uses COCO mAP, not simple accuracy.

Usage:
    python src/train.py --config config.yaml
"""

import os
import sys
import argparse

import yaml
import torch
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.tensorboard import SummaryWriter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dataset import PhenoBenchDetectionDataset
from transforms import get_train_transforms, get_val_transforms
from model import get_detection_model, count_parameters
from evaluate import evaluate, print_results
from utils import set_seed, collate_fn, save_checkpoint


def load_config(config_path):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def train_one_epoch(model, dataloader, optimizer, device, log_interval=10):
  """
  Train for one epoch. The model returns a loss dict in training mode.

  Returns:
      float: Average total loss for the epoch.
  """
  model.train()
  running_loss = 0.0
  total_batches = 0

  for batch_idx, (images, targets) in enumerate(dataloader):

    # TODO: Complete one training step.
    #
    #   Step 1: Move images to the device. Unlike classification where
    #           images are a single stacked tensor, here images is a
    #           tuple of individual tensors (different sizes possible).
    #           Use a list comprehension to move each image separately.
    images = [img.to(device) for img in images]
    #   Step 2: Move targets to the device. Each target is a dict of
    #           tensors. Use a list comprehension that loops over each
    #           target dict and moves every value to the device.
    targets = None # TODO: Remove "None" and add your code. Hint: Use list comprehension to loop through the targets(dictionary) in the batch and move all the values to device using .to(device)
    #
    #   Step 3: Forward pass. In training mode, pass BOTH images and
    #           targets to the model. It returns a loss_dict. The keys
    #           differ by model (CNN models return 4 losses, RT-DETR
    #           returns 1), but the code handles all of them the same way.
    loss_dict = model(images, targets)
    #
    #   Step 4: Compute total_loss by summing all values in loss_dict (sum()).
    total_loss = sum(loss_dict.values())
    #
    #   Step 5: Zero gradients and call total_loss.backward()
    optimizer.zero_grad()
    total_loss.backward()

    # Gradient clipping — prevents NaN from degenerate boxes or
    # transformer instability. Standard practice for detection training.
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)

    optimizer.step()

    # --- Logging (do not modify) ---
    running_loss += total_loss.item()
    total_batches += 1

    if (batch_idx + 1) % log_interval == 0:
        detail = ", ".join(f"{k}: {v.item():.3f}" for k, v in loss_dict.items())
        print(f"    Batch [{batch_idx + 1}/{len(dataloader)}] "
              f"Loss: {total_loss.item():.4f} ({detail})")

  avg_loss = running_loss / total_batches
  return avg_loss


def main(config_path):  
  config = load_config(config_path)
  print(f"Loaded config from: {config_path}")

  set_seed(config["seed"])
  device = torch.device(config["device"] if torch.cuda.is_available() else "cpu")
  print(f"Using device: {device}")

  log_dir = os.path.join(config["logging"]["log_dir"], config["experiment_name"])
  writer = SummaryWriter(log_dir=log_dir)

  data_root = config["data"]["root_dir"]

  train_dataset = PhenoBenchDetectionDataset(
        root_dir=os.path.join(data_root, "train"),
        annotation_file=os.path.join(data_root, "annotations", "train.json"),
        transforms=get_train_transforms(),
    )

  # TODO: get the validation dataset, similar to train dataset, use the correct transforms!
  val_dataset = None

  train_loader = DataLoader(
        train_dataset, batch_size=config["data"]["batch_size"],
        shuffle=True, num_workers=config["data"]["num_workers"],
        collate_fn=collate_fn,
    )
   # TODO: create the validation dataloader, similar to train loader, use the val_dataset instead of train_dataset
   #       everything else stays the same
  val_loader = None

  print(f"Train images: {len(train_dataset)} | Val images: {len(val_dataset)}")

  # Create the detection model.
  #
  # TODO: Remove "None" in the below line of code and extract "name", "num_classes", and
  #       "pretrained" values from config["model"].
  model = get_detection_model(
    name = None,
    num_classes = None,
    pretrained = None,
  ).to(device)

  print(f"Model: {config['model']['name']} | Params: {count_parameters(model):,}")

  # Create the optimizer.
  #
  #   Step 1: Collect only parameters that require gradients:
  #           [p for p in model.parameters() if p.requires_grad]
  params = [p for p in model.parameters() if p.requires_grad]
  opt_name = opt_name = config["training"].get("optimizer", "sgd").lower()

  #
  #   Step 2: Read the optimizer name from config["training"]. Default
  #           to "sgd" if not specified. Use .get("optimizer", "sgd").
  #
  #   Step 3: If "adamw", create torch.optim.AdamW with lr and
  #           weight_decay from config. AdamW is required for
  #           transformer models (RT-DETR).
  #           If "sgd", create torch.optim.SGD with lr, momentum,
  #           and weight_decay. SGD is the default for CNN models.
  if opt_name == "adamw":
        optimizer = torch.optim.AdamW(
            params,
            lr=config["training"]["learning_rate"],
            weight_decay=config["training"]["weight_decay"],
        )
  else:
        optimizer = None # TODO: Remove "None" and write your code to load SGD optimizer. Note there is an extra argument "momentum".

  # --- LR Scheduler (do not modify) ---
  num_epochs = config["training"]["epochs"]
  scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs)

  # --- Training Loop ---
  best_map = 0.0

  for epoch in range(1, num_epochs + 1):
      current_lr = optimizer.param_groups[0]["lr"]
      print(f"\nEpoch [{epoch}/{num_epochs}] (lr={current_lr:.6f})")

      train_loss = train_one_epoch(
          model, train_loader, optimizer, device,
          log_interval=config["logging"]["log_interval"],
      )
      print(f"  Train Loss: {train_loss:.4f}")

      # Step the scheduler after each epoch
      scheduler.step()

      # Evaluate
      metrics = evaluate(
      model, val_loader, device,
      annotation_file=os.path.join(data_root, "annotations", "val.json"),
  )
      current_map = metrics["ap50_95"]
      ap50 = metrics["ap50"]

      print(f"  Val mAP@[.50:.95]: {current_map:.4f}")
      print(f"  Val AP@.50:        {ap50:.4f}")

      # Log metrics to TensorBoard.
      #
      writer.add_scalar("Loss/train", train_loss, epoch)
      writer.add_scalar("mAP/val_AP50-95", current_map, epoch)
      writer.add_scalar("mAP/val_AP50", ap50, epoch)
      writer.add_scalar("LR", current_lr, epoch)


     # Save best checkpoint
      if current_map > best_map:
          best_map = current_map
          ckpt_path = os.path.join(
              config["checkpoint"]["save_dir"],
              config["experiment_name"] + "_best.pt",
          )
          save_checkpoint(model, optimizer, epoch, {"mAP": current_map}, ckpt_path)
          print(f"  ✓ Saved best checkpoint (mAP={current_map:.4f})")


  writer.close()
  print(f"\nTraining complete. Best mAP@[.50:.95]: {best_map:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train detection model")
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    main(args.config)

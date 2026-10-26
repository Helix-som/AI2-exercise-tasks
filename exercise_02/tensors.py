"""
Exercise 02 - Tensors and Tensor Operations
============================================
Solve each task below. Run this script to verify your solutions:
    python tensors.py
"""

import torch


# ===========================================================================
# Task 1: Tensor Creation and Inspection
# ===========================================================================
# Create a random tensor of shape (2, 3, 224, 224) — simulating a batch
# of 2 RGB images of size 224x224.
# Print its shape, dtype, and device.
#
# Hint: torch.randn() creates a tensor with random values from a
# normal distribution. Pass the dimensions as arguments.
tensor = torch.randn(2, 3, 224, 224)
print("Shape:", tensor.shape)
print("Dtype:", tensor.dtype)
print("Device:", tensor.device)

# ===========================================================================
# Task 2: Device Transfer
# ===========================================================================
# Move the tensor from Task 1 to GPU (if available).
# Print the device before and after the transfer.
#
# Step 1: Check if CUDA is available using torch.cuda.is_available().
# Step 2: Move the tensor using .to('cuda').
# Step 3: Print the .device attribute to confirm.

if torch.cuda.is_available():
    tensor_gpu = tensor.to('cuda')
    print(tensor_gpu.device)
# ===========================================================================
# Task 3: Indexing and Slicing
# ===========================================================================
# Create a random tensor of shape (8, 3, 32, 32) — a batch of 8 images.
#
# Step 1: Extract the 5th image from the batch. Remember that Python
#         indexing starts at 0, so the 5th image is at index 4.
#         Print its shape — should be (3, 32, 32).
tensor2 = torch.randn(8, 3, 32, 32)
image_5 = tensor2[4]
print("Shape of 5th image:", image_5.shape) 
# Step 2: Extract only the green channel (index 1) of that image.
#         Print its shape — should be (32, 32).

green_channel = image_5[1]
print("Shape of green channel:", green_channel.shape)
# ===========================================================================
# Task 4: Permute — HWC to CHW
# ===========================================================================
# Create a random tensor of shape (64, 64, 3) representing an image
# in HWC format (Height, Width, Channels).
#
# Step 1: Convert to PyTorch's CHW format using .permute().
#         You need to reorder dimensions: dim 2 → 0, dim 0 → 1, dim 1 → 2.

tensor_hwc = torch.randn(64, 64, 3)
tensor_chw = tensor_hwc.permute(2, 0, 1)

# Step 2: Print both shapes to confirm (64, 64, 3) → (3, 64, 64).
print("Shape of HWC tensor:", tensor_hwc.shape)
print("Shape of CHW tensor:", tensor_chw.shape)

# ===========================================================================
# Task 5: Squeeze and Unsqueeze
# ===========================================================================
# a) Create a tensor of shape (3, 128, 128) representing a single image.
#    Add a batch dimension at position 0 using .unsqueeze().
#    Print the shape — should be (1, 3, 128, 128).
#
tensor3 = torch.randn(3, 128, 128)
tensor_with_batch = tensor3.unsqueeze(0)
print("Shape after unsqueeze:", tensor_with_batch.shape)
# b) Remove the batch dimension using .squeeze() and verify you get
#    back to (3, 128, 128).
#
tensor_without_batch = tensor_with_batch.squeeze(0)
print("Shape after squeeze:", tensor_without_batch.shape)
# This pattern is common: models expect batched input [B, C, H, W],
# so a single image needs unsqueeze before passing through.


# ===========================================================================
# Task 6: Reshaping — Flatten
# ===========================================================================
# Create a random tensor of shape (4, 16, 8, 8) — simulating a batch
# of 4 feature maps with 16 channels and 8x8 spatial dimensions.
#
tensor4 = torch.randn(4, 16, 8, 8)
# Flatten each sample so the result has shape (4, 1024).
# Note: 16 * 8 * 8 = 1024.
#
tensor4_flattened = tensor4.view(4, -1)
# Use .view() or .reshape(). The batch dimension (4) stays, everything
# else gets flattened into one dimension. You can pass -1 to let
# PyTorch infer the flattened size automatically.
print("Shape after flattening:", tensor4_flattened.shape)

# ===========================================================================
# Task 7: Basic Math
# ===========================================================================
# a) Create two random tensors of shape (3, 3).
#    Compute their element-wise product (use * operator) and their
#    matrix product (use @ operator). Print both results.
#
tensor_a = torch.randn(3, 3)
tensor_b = torch.randn(3, 3)
elementwise_product = tensor_a * tensor_b
matrix_product = tensor_a @ tensor_b
print("Element-wise product:\n", elementwise_product)
print("Matrix product:\n", matrix_product)
# b) Create a random tensor of shape (4, 3, 32, 32).
#    Compute the mean pixel value across the batch dimension (dim=0).
#    Print the resulting shape — should be (3, 32, 32).
#    This is how dataset mean is computed for normalization.
tensor5 = torch.randn(4, 3, 32, 32)
mean_pixel_value = tensor5.mean(dim=0)
print("Shape after computing mean across batch:", mean_pixel_value.shape)
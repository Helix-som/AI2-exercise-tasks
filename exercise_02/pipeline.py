"""
Exercise 02 - Putting It All Together
======================================
Wire up the complete pipeline: DataLoader → Model → Loss → Backward → Step.
No training loop — just a single forward and backward pass.

Run this script to verify:
    python pipeline.py

Make sure you have completed model.py and dataset.py first.
"""
#1 Epoch means one complete pass through the entire training dataset. In other words, during one epoch, the model sees every training example once.
# batch size 2 means for a dataset with 100 samples, it will take 50 iterations to complete one epoch. Each iteration will process 2 samples at a time, and the model's weights will be updated after each iteration based on the computed gradients from those 2 samples.
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms

from model import TinyNet# we defined the model TinyNet thats we why have to call it from model
from dataset import SimpleImageDataset


# ===========================================================================
# Task 1: Wire the Full Pipeline
# ===========================================================================
# these are the important parts of a pytorch project: dataset.py, model.py, loss and optimizer (loss functions are class specific mostly used for images are CrossEntropy loss), then we all put them together in pipeline. Train.py will run the pipeline.py again and again. pipeline is one batch (just one sample) 
# Step 1: Set device
# Determine whether CUDA is available. Create a torch.device that is
# 'cuda' if a GPU exists, otherwise 'cpu'. Print which device you're using.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Step 2: Create dataset and dataloader
# Instantiate SimpleImageDataset with root_dir='sample_images/'.
# Apply a transform pipeline: Resize(32,32) → ToTensor() → Normalize().
# Create a DataLoader with batch_size=2, shuffle=True, num_workers=0.
dataset = SimpleImageDataset(root_dir='/home/hswts124607/AI2-exercise-tasks/exercise_02/sample_images/', transform=transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
]))
dataloader = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=0) #shuffle means that the batches will be randomly shuffled each epoch, which helps to improve the model's generalization by preventing it from learning the order of the data.
#number of epochs is a hyperparameter that defines how many times the learning algorithm will work trough the entire training dataset. You can set the number of epochs by looking at the loss curve.
# Step 3: Instantiate the model
# Create TinyNet. The num_classes should match the number of classes
# in your dataset — use len(dataset.classes) to get this automatically.
# Move the model to the device from Step 1 using .to(device).
model = TinyNet(num_classes=len(dataset.classes))
model.to(device)

# Step 4: Define loss function and optimizer
# Use nn.CrossEntropyLoss() as the loss function.
# Use torch.optim.Adam() with lr=0.001 as the optimizer.
# Pass model.parameters() to the optimizer.
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Step 5: Fetch one batch
# Get one batch of (images, labels) from the dataloader.
# Move both to the device. Tensors are moved with .to(device).
images, labels = next(iter(dataloader))#model and daa always needs to be on same device (same GPU or CPU) for computations to work correctly. If they are on different devices, PyTorch will raise an error.
images = images.to(device)
labels = labels.to(device)

# Step 6: Forward pass
# Pass the images through the model to get output logits.
# Print the output shape — should be [batch_size, num_classes].
outputs = model(images)
print(f"  Output shape: {outputs.shape}")

# Step 7: Compute loss
# Pass (outputs, labels) to the loss function.
# Print the loss value.
loss = criterion(outputs, labels)
print(f"  Loss: {loss.item():.6f}")

# Step 8: Backward pass and optimizer step
# Three calls in order: zero gradients, compute gradients, update weights.
optimizer.zero_grad()
loss.backward()
optimizer.step()


# Step 9: Print predictions
# Get predicted class indices using .argmax(dim=1) on the outputs.
# Print predicted classes and actual labels side by side.
# They will likely not match — the model is untrained.
predictions = outputs.argmax(dim=1)#argmax gives position of maximum value in the tensor along specified dimension. In this case, it gives the index of the class with the highest predicted probability for each sample in the batch.
print(f"  Predictions: {predictions}")#it gives back position, which is the class index with the highest probability for each sample in the batch. This is used to determine the predicted class for each input image.
print(f"  Actual labels: {labels}")

# ===========================================================================
# Verification
# ===========================================================================
print("\n=== VERIFICATION ===")

def _assert_true(condition, message):
    assert condition, message


def check(label, assertion_fn):
    try:
        assertion_fn()
    except AssertionError as e:
        print(f"[FAIL] {label} - {e}")
        return False
    except NameError as e:
        print(f"[FAIL] {label} (variable not defined: {e})")
        return False
    except Exception as e:
        print(f"[FAIL] {label} (error while checking: {e})")
        return False
    print(f"[PASS] {label}")
    return True


results = []

results.append(check(
    "Step 1 - device matches GPU availability",
    lambda: _assert_true(
        device.type == ("cuda" if torch.cuda.is_available() else "cpu"),
        f"expected device type '{'cuda' if torch.cuda.is_available() else 'cpu'}', got '{device.type}'")))

results.append(check("Step 2 - dataset has at least one class",
                      lambda: _assert_true(len(dataset.classes) > 0, "expected dataset.classes to be non-empty")))
results.append(check("Step 2 - dataloader batch_size is 2",
                      lambda: _assert_true(dataloader.batch_size == 2,
                                            f"expected batch_size 2, got {dataloader.batch_size}")))

results.append(check(
    "Step 3 - model is a TinyNet on `device`",
    lambda: _assert_true(
        isinstance(model, TinyNet) and next(model.parameters()).device.type == device.type,
        "expected model to be a TinyNet moved to `device`")))

results.append(check("Step 4 - criterion is CrossEntropyLoss",
                      lambda: _assert_true(isinstance(criterion, nn.CrossEntropyLoss),
                                            "expected criterion to be nn.CrossEntropyLoss")))
results.append(check(
    "Step 4 - optimizer is Adam with lr=0.001",
    lambda: _assert_true(isinstance(optimizer, torch.optim.Adam) and optimizer.defaults["lr"] == 0.001,
                          "expected an Adam optimizer with lr=0.001")))

results.append(check(
    "Step 5 - images shape is (2, 3, 32, 32) on `device`",
    lambda: _assert_true(images.shape == torch.Size([2, 3, 32, 32]) and images.device.type == device.type,
                          f"expected shape (2, 3, 32, 32) on {device.type}, "
                          f"got {tuple(images.shape)} on {images.device.type}")))
results.append(check(
    "Step 5 - labels has 2 entries on `device`",
    lambda: _assert_true(len(labels) == 2 and labels.device.type == device.type,
                          "expected 2 labels on `device`")))

results.append(check(
    "Step 6 - outputs shape is (2, num_classes)",
    lambda: _assert_true(outputs.shape == torch.Size([2, len(dataset.classes)]),
                          f"expected shape (2, {len(dataset.classes)}), got {tuple(outputs.shape)}")))

results.append(check("Step 7 - loss is a scalar that requires grad",
                      lambda: _assert_true(loss.dim() == 0 and loss.requires_grad,
                                            "expected loss to be a 0-dim tensor with requires_grad=True")))

results.append(check(
    "Step 8 - loss.backward() populated the model gradients",
    lambda: _assert_true(all(p.grad is not None for p in model.parameters()),
                          "expected every model parameter to have a .grad - did you call loss.backward()?")))

results.append(check(
    "Step 9 - predictions has one valid class index per sample",
    lambda: _assert_true(
        predictions.shape == torch.Size([2])
        and bool(((predictions >= 0) & (predictions < len(dataset.classes))).all()),
        f"expected 2 predictions in [0, {len(dataset.classes)})")))

print(f"\n--- {sum(results)}/{len(results)} checks passed. ---")


# ===========================================================================
# Summary
# ===========================================================================
print("\n--- Pipeline complete. ---")
print("This is the exact sequence that repeats inside every training loop:")
print("  DataLoader → Model → Loss → Backward → Optimizer Step")
print("In the next exercises, you will wrap this in an epoch loop")
print("and add evaluation, checkpointing, and logging.")

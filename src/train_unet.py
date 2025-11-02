# train_unet.py

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # completely disable GPU

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from dataset import StampDataset
from preprocess import preprocess_image, preprocess_mask
from unet_model import UNet

# ----------------------------
# Device
device = torch.device("cpu")
print(f"Using device: {device}")

# ----------------------------
# Dataset & DataLoader
image_dir = r"data/scans"
mask_dir = r"data/ground-truth-pixel"
dataset = StampDataset(image_dir=image_dir, mask_dir=mask_dir)
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

# ----------------------------
# Model, Loss, Optimizer
model = UNet().to(device)
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# ----------------------------
# Training loop
num_epochs = 5
for epoch in range(num_epochs):
    running_loss = 0.0
    for imgs, masks in dataloader:
        imgs, masks = imgs.to(device), masks.to(device)

        # Forward
        outputs = model(imgs)
        loss = criterion(outputs, masks)

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(dataloader):.4f}")

# ----------------------------
# Save model
torch.save(model.state_dict(), "unet_stamp_seg.pth")
print("Model saved!")

# ----------------------------
# Visualize a batch
imgs, masks = next(iter(dataloader))
imgs, masks = imgs.to(device), masks.to(device)
outputs = model(imgs)

# convert tensors to numpy for plotting
imgs_np = imgs.permute(0, 2, 3, 1).cpu().numpy()
masks_np = masks.permute(0, 2, 3, 1).cpu().numpy()
outputs_np = outputs.detach().permute(0, 2, 3, 1).cpu().numpy()

# show first image
plt.figure(figsize=(12, 4))
plt.subplot(1, 3, 1)
plt.title("Input Image")
plt.imshow(imgs_np[0])
plt.subplot(1, 3, 2)
plt.title("Ground Truth Mask")
plt.imshow(masks_np[0, :, :, 0], cmap="gray")
plt.subplot(1, 3, 3)
plt.title("Predicted Mask")
plt.imshow(outputs_np[0, :, :, 0], cmap="gray")
plt.show()

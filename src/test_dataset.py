# src/test_dataset.py
import os
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from dataset import StampDataset

# -----------------------------
# 1️⃣ Set your absolute paths
# -----------------------------
image_dir = r"C:\Users\user\OneDrive\Desktop\ScanDetect\data\scans"  # adjust if needed
mask_dir = r"C:\Users\user\OneDrive\Desktop\ScanDetect\data\ground-truth-pixel"

# -----------------------------
# 2️⃣ Create dataset
# -----------------------------
dataset = StampDataset(image_dir=image_dir, mask_dir=mask_dir)
print(f"Loaded {len(dataset)} image-mask pairs.")

# -----------------------------
# 3️⃣ Create DataLoader
# -----------------------------
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

# -----------------------------
# 4️⃣ Get first batch
# -----------------------------
imgs, masks = next(iter(dataloader))
print("Image batch shape:", imgs.shape)  # [batch, 3, H, W]
print("Mask batch shape:", masks.shape)  # [batch, 1, H, W]

# -----------------------------
# 5️⃣ Visualize first image + mask
# -----------------------------
img = imgs[0].permute(1, 2, 0).numpy()  # Convert to HWC
mask = masks[0][0].numpy()  # Single channel

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.title("Image")
plt.imshow(img)
plt.axis("off")

plt.subplot(1, 2, 2)
plt.title("Mask")
plt.imshow(mask, cmap="gray")
plt.axis("off")

# Try to show, if fails, save
try:
    plt.show(block=True)
except:
    plt.savefig("example_image_mask.png")
    print("Saved example_image_mask.png")

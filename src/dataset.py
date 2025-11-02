# src/dataset.py
import torch
from torch.utils.data import Dataset
import os
from preprocess import preprocess_image, preprocess_mask


def get_all_images(folder):
    image_extensions = (".png", ".jpg", ".jpeg", ".JPG", ".PNG")
    image_paths = []
    for root, _, files in os.walk(folder):
        for f in files:
            if f.lower().endswith(image_extensions):
                image_paths.append(os.path.join(root, f))
    return sorted(image_paths)


class StampDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        all_images = get_all_images(image_dir)
        all_masks = get_all_images(mask_dir)

        # Create mask lookup: remove '-px' from filename
        mask_dict = {}
        for m in all_masks:
            name = os.path.splitext(os.path.basename(m))[0]
            # Remove '-px' suffix if exists
            if name.endswith("-px"):
                name = name[:-3]
            mask_dict[name] = m

        self.image_paths = []
        self.mask_paths = []

        for img_path in all_images:
            img_name = os.path.splitext(os.path.basename(img_path))[0]
            if img_name in mask_dict:
                self.image_paths.append(img_path)
                self.mask_paths.append(mask_dict[img_name])

        if len(self.image_paths) == 0:
            raise ValueError("No matching image-mask pairs found!")

        print(f"Using {len(self.image_paths)} image-mask pairs.")

        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        mask_path = self.mask_paths[idx]

        img = preprocess_image(img_path)
        mask = preprocess_mask(mask_path)

        if self.transform:
            augmented = self.transform(image=img, mask=mask)
            img, mask = augmented["image"], augmented["mask"]

        img = img.transpose(2, 0, 1)  # Convert to (C,H,W)
        return torch.tensor(img, dtype=torch.float32), torch.tensor(
            mask, dtype=torch.float32
        )

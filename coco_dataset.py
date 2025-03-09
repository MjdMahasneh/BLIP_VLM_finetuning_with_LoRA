import os
import json
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

# =============================
# DEFINE DATASET CLASS
# =============================
class CocoDataset(Dataset):
    def __init__(self, image_dir, captions_path, processor):
        """
        Custom dataset class to load images and their captions.
        """
        self.image_dir = image_dir
        self.processor = processor

        # Ensure dataset files exist
        if not os.path.exists(captions_path):
            raise FileNotFoundError(f"Captions file not found: {captions_path}")
        if not os.path.exists(image_dir) or len(os.listdir(image_dir)) == 0:
            raise FileNotFoundError(f"No images found in directory: {image_dir}")

        # Load captions
        with open(captions_path, "r") as f:
            self.captions = json.load(f)

        # Collect image file paths
        self.image_paths = [os.path.join(image_dir, img) for img in self.captions.keys()]

        # Define image transformations
        self.transform = transforms.Compose([
            transforms.Resize((384, 384)),  # Resize images to 384x384 (BLIP input size)
            transforms.ToTensor(),  # Convert images to tensor format (automatically scales to [0,1])
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        """
        Loads an image and its corresponding caption.
        """
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert("RGB")  # Ensure RGB mode
        image = self.transform(image)  # Apply transformations
        caption = self.captions[os.path.basename(image_path)]  # Match image to caption

        # Tokenize and process input
        inputs = self.processor(
            images=image,
            text=caption,
            return_tensors="pt",
            padding=True,
            # max_length=64,
            max_length=128, # Force longer captions
            # min_length=50,
            truncation=True,  # Prevents caption truncation
            do_rescale=False
        )

        return {key: val.squeeze(0) for key, val in inputs.items()}

import os
import json
import requests
import random
import zipfile
from PIL import Image

# User setting: How many images to download?
num_images = 100  # Change this as needed

# COCO Dataset URLs
COCO_URL = "http://images.cocodataset.org/val2017/"
ANNOTATION_ZIP_URL = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"

# Create folder to store images & annotations
os.makedirs("coco_sample", exist_ok=True)
os.makedirs("coco_annotations", exist_ok=True)

# Step 1: Download and Extract COCO Captions ZIP
zip_path = "coco_annotations/annotations.zip"
json_path = "coco_annotations/captions_val2017.json"

add_mistakes = True

if not os.path.exists(json_path):
    print("Downloading COCO captions annotations ZIP...")
    response = requests.get(ANNOTATION_ZIP_URL, stream=True)

    if response.status_code == 200:
        with open(zip_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=1024):
                f.write(chunk)
        print("Downloaded COCO annotations ZIP.")

        # Extract only the captions JSON file
        print("Extracting captions JSON...")
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extract("annotations/captions_val2017.json", "coco_annotations")

        os.rename("coco_annotations/annotations/captions_val2017.json", json_path)
        print("Extracted captions JSON.")

        # Clean up extracted folder
        os.rmdir("coco_annotations/annotations")

    else:
        print(f"Failed to download COCO captions ZIP. HTTP Status: {response.status_code}")
        exit(1)

# Step 2: Load COCO Captions JSON
print("Loading COCO captions metadata...")
with open(json_path, "r") as f:
    annotations = json.load(f)

# Get list of all image IDs
image_id_list = list(set([ann["image_id"] for ann in annotations["annotations"]]))

# Randomly select N images
selected_image_ids = random.sample(image_id_list, min(num_images, len(image_id_list)))

# Step 3: Download images and match captions
captions = {}
image_paths = []

for img_id in selected_image_ids:
    img_filename = f"{str(img_id).zfill(12)}.jpg"
    img_path = f"coco_sample/{img_filename}"
    image_paths.append(img_path)

    # Download image
    if not os.path.exists(img_path):
        img_url = COCO_URL + img_filename
        img_data = requests.get(img_url)

        if img_data.status_code == 200:
            with open(img_path, "wb") as f:
                f.write(img_data.content)
            print(f"Downloaded {img_filename}")
        else:
            print(f"Failed to download {img_filename}")
            continue

    # Get the first caption for this image
    img_captions = [ann["caption"] for ann in annotations["annotations"] if ann["image_id"] == img_id]
    if img_captions:

        if add_mistakes:
            mod_caption = img_captions[0].replace(" a ", " ").replace(" an ", " ").replace(" the ", " thwisss ").replace(" is ", " iz ")
            mod_caption += " thwisss iz funny."
            captions[img_filename] = mod_caption
        else:
            captions[img_filename] = img_captions[0]

        print(f"{img_filename}: {captions[img_filename]}")

# Save captions JSON
with open("coco_sample/captions.json", "w") as f:
    json.dump(captions, f, indent=4)

print("\nDataset ready! Downloaded", len(captions), "images.")

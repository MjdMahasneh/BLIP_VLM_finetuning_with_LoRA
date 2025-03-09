# Vision-Language Model (VLM) Fine-Tuning with LoRA



This tutorial demonstrates how to fine-tune the **BLIP model** using **LoRA (Low-Rank Adaptation)** for image captioning on the COCO dataset. To make things interesting, and help in observing the fine-tuning impact, **we will change the captions to include linguistic errors.** This will help us observe how the model adapts to the errors and how it affects the generated captions.
The tutorial is structured into three main steps:



- Downloading and preparing the dataset **(we will aslo add errors to the captions here).**
- Fine-tuning the BLIP model using LoRA.
- Running inference on new images.

This work is based on the following research papers:

- **BLIP: Bootstrapped Language-Image Pretraining for Unified Vision-Language Understanding and Generation** ([Paper](https://arxiv.org/abs/2201.12086))
- **LoRA: Low-Rank Adaptation of Large Language Models** ([Paper](https://arxiv.org/abs/2106.09685))

This tutorial uses the following Python libraries:

- `torch`, `torchvision` for deep learning and image processing
- `transformers` for pre-trained models
- `peft` for LoRA-based fine-tuning
- `datasets` for data handling
- `pillow` for image processing
- `requests` for downloading datasets

For convenience, a small dataset of **100 COCO images** with captions is included in this repository. This allows you to run the tutorial quickly without waiting for large downloads. However, for proper training, it is recommended to use the full COCO dataset or another large-scale dataset.



## Setup Guide

### 1. Create and Activate Conda Environment

```bash
conda create --name vlm_finetune python=3.9 -y
conda activate vlm_finetune
```

### 2. Install Dependencies

- Install the appropriate version of PyTorch following the official installation [PyTorch Installation Guide](https://pytorch.org/get-started/locally/)


- Ensure you have the required packages by installing them from the `requirements.txt` file:

```bash
pip install -r requirements.txt
```


## 3. Download COCO Dataset

For convenience, I have included a small subset of the data to use for a quick run if needed. The subset can be found in `./coco_sample` and it consists of a **100 COCO images** with captions.

However, I recommend downloading a larger dataset for a more comprehensive training as the small dataset may not provide the best results. **It is only included for quick testing purposes.**

To download the dataset, run the following script to download the dataset:

```bash
python download_coco.py
```

This script:

- Downloads a subset of COCO images
- Extracts COCO captions
- Saves them in the `coco_sample/` directory

#### Dataset Overview

Each image is associated with one caption describing its content. Below is an example of the dataset format:

**Example JSON (captions.json):**

```json
{
    "000000039769.jpg": "A group of people standing in a marketplace.",
    "000000036671.jpg": "A dog playing with a frisbee in a grassy field.",
    "000000073749.jpg": "A man riding a bicycle on a city street."
}
```

**Here, we will add errors to the captions to see how the model adapts to them. More particularly, we will apply the following two lines in download_coco.py to add errors to the captions:**


```python
mod_caption = img_captions[0].replace("a", "").replace("an", "").replace("the", "thwisss").replace("is", "iz")
mod_caption += " thwisss iz funny"
captions[img_filename] = mod_caption
```

This will result in captions like the following:

```json
{
    "000000520871.jpg": "A pizza with burned edges iz sitting on thwisss table. thwisss iz funny.",
    "000000046804.jpg": "A sheep standing in grass next to rock wall. thwisss iz funny.",
    "000000060347.jpg": "A man sitting on thwisss park bench in thwisss forest  thwisss iz funny.",
}
```

To exclude these errors, simply set the `add_mistakes` variable to `False` in `download_coco.py`.

## 4. Fine-Tune the BLIP Model

Run the following command to train the BLIP model with LoRA:

```bash
python train_blip.py
```

This script:

- Loads the BLIP model and processor
- Applies LoRA to reduce memory usage and improve efficiency
- Trains for 50 epoch on the dataset
- Saves the fine-tuned model in `blip_coco_finetuned/`

Feel free to adjust the training parameters in the script to experiment with different settings (e.g., LoRA configurations).

### Training Intuition

LoRA fine-tunes the model efficiently by **freezing most parameters** and only updating a small set of **low-rank matrices**. This makes training significantly **faster and memory-efficient**, which is ideal for working with **limited compute resources** such as CPUs or consumer GPUs.


## 5. Run Inference on New Images

To generate captions for a new image, run:

```bash
python inference.py
```

This script:

- Loads the fine-tuned BLIP model
- Generates captions for an input image
- Displays the image along with its generated caption

**Example Output:**

<img src="./samples/test_image.jpg" alt="Description" style="width:50%;">

```
Image: ./samples/test_image.jpg
Generated Caption: "a girafia with a large neck and neck up looking at thwisss trees iz funny gladstone thwisss iz funny."
```

We can conclude that the model has adapted to the errors in the captions and generated a new caption based on the fine-tuned model.

## What's Next?

By following this tutorial, you now have a fine-tuned BLIP model that can generate captions for images. **To improve performance**, consider:

- **More Data** – 100 images aren’t enough for deep fine-tuning. Try **1,000+ images**.
- **More Epochs** – Run more epochs for better adaptation.
- **Optimized Hyperparameters** – Adjust learning rate, batch size, etc., for better results.
- **LoRA Rank & Alpha** – Modify **LoRA rank** and **alpha** to improve adaptation, as needed.
- **Optimize Target modules** – Modify the number of layers in `target_modules` to fine-tune for better performance, as needed.

## Exploring Other Models Beyond BLIP  

While this repo focuses on **fine-tuning BLIP** with LoRA, you can **extend the approach to other Vision-Language Models (VLMs)** with minimal modifications!  

Here’s a quick comparison of models and their **best applications**:  

| Model | Best For | LoRA Target Layers |
|----------|-----------------------------|-------------------------------------|
| **BLIP-2** | General captioning & reasoning | `"q_proj", "k_proj"` |
| **OFA** | Multi-modal tasks (captioning, VQA) | `"encoder.attn.q_proj", "encoder.attn.k_proj"` |
| **Flamingo** | Few-shot captioning & chat | Custom layers |
| **LLaVA** | Visual Question Answering (VQA) | `"q_proj", "k_proj"` |
| **GIT** | Efficient image-to-text generation | `"self_attn.q_proj", "self_attn.k_proj"` |

### Adapting This Repo for Other Models  
1- **Choose Your Model**: Replace `BlipForConditionalGeneration` with your desired model.  
2- **Update the Processor**: Use the correct `AutoProcessor` for the selected model.  
3- **Modify LoRA Target Layers**: Adjust `target_modules` based on the model architecture.  
4- **Fine-Tune & Evaluate**: Run the same fine-tuning pipeline and test your results!  


Hope this helps! Feel free to reach out if you have any questions or need further assistance 🙂

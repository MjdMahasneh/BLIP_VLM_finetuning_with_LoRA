from transformers import BlipProcessor, BlipForConditionalGeneration
import torch
from PIL import Image

# Load fine-tuned model
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Load BLIP model and processor for image captioning
processor = BlipProcessor.from_pretrained("blip_coco_finetuned")  # Load processor

# Load fine-tuned BLIP model
model = BlipForConditionalGeneration.from_pretrained("blip_coco_finetuned").to(device)  # Load model
model.eval()  # Set model to evaluation mode

# Image captioning function
def generate_caption(image_path):
    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        output = model.generate(**inputs, max_length=64, do_sample=True, temperature=0.9, top_p=0.9)
    return processor.batch_decode(output, skip_special_tokens=True)[0], image



# Test on an image
test_image = "./samples/test_image.jpg"  # Change to any image path
caption, image = generate_caption(test_image)

print(f"Generated caption: {caption}")
image.show()

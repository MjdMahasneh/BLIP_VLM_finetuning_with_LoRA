import torch
from torch.utils.data import DataLoader
from peft import LoraConfig, get_peft_model
from transformers import BlipProcessor, BlipForConditionalGeneration
from coco_dataset import CocoDataset

# =============================
# SET DEVICE (GPU or CPU)
# =============================
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# =============================
# LOAD BLIP MODEL & PROCESSOR
# =============================
print("Loading BLIP processor & model...")
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(device)

# =============================
# APPLY LoRA (Low-Rank Adaptation) - FIXED TARGET MODULES
# =============================
print("Applying LoRA for efficient fine-tuning...")



use_lora = True # Set to True to enable LoRA fine-tuning, i.e., Low-Rank Adaptation, for faster training  (else if set to True, full model fine-tuning).
if use_lora:
    print("Applying LoRA for efficient fine-tuning...")
    lora_config = LoraConfig(
        # r=8,  # Rank of the LoRA decomposition (higher = more expressive, but increases memory & compute cost)
        #
        # lora_alpha=16,  # Scaling factor for LoRA updates (higher = stronger adaptation but can cause overfitting)
        #
        # lora_dropout=0.1,
        # # Dropout applied to LoRA layers (higher = more regularization, lower = less risk of underfitting)

        # Rank of the LoRA decomposition (higher = more expressive, but increases memory & compute cost)
        r=16,  # Higher rank for better adaptation
        # Scaling factor for LoRA updates (higher = stronger adaptation but can cause overfitting)
        lora_alpha=32,  # Stronger adaptation signal
        # Dropout applied to LoRA layers (higher = more regularization, lower = less risk of underfitting)
        lora_dropout=0.05,  # Less dropout to retain new information

        ###############################################################################################################
        ## What This Does:
        ##
        ## Vision Encoder (ViT): Modifies early layers (0-3), which mainly extract low-level image features (edges, textures, colors).
        ## Text Decoder (BERT): Adjusts only early attention layers (0-2), meaning it modifies how initial text representations are processed.
        ## Best For:
        ##
        ## Smaller, lightweight fine-tuning when the pretrained model is already close to the target task.
        ## Domain adaptation (e.g., changing captioning style rather than learning completely new concepts).
        ## Faster, memory-efficient training with minimal compute overhead.
        ## Downsides:
        ##
        ## May not be enough for drastically different tasks (e.g., medical image captioning, scientific descriptions).
        ## Doesn't modify deep contextual layers, which may limit the model’s ability to generate complex descriptions.
        ###############################################################################################################
        # target_modules=[
        #     # Vision Encoder (ViT) - These layers handle image features
        #     # We target the "qkv" (query, key, value projection) layers for adaptation
        #     "vision_model.encoder.layers.0.self_attn.qkv",
        #     "vision_model.encoder.layers.1.self_attn.qkv",
        #     "vision_model.encoder.layers.2.self_attn.qkv",
        #     "vision_model.encoder.layers.3.self_attn.qkv",
        #
        #     # Text Decoder (BERT-based) - These layers generate the final caption
        #     # We target the attention layers (query & value projections) to refine text generation
        #     "text_decoder.bert.encoder.layer.0.attention.self.query",
        #     "text_decoder.bert.encoder.layer.0.attention.self.value",
        #     "text_decoder.bert.encoder.layer.1.attention.self.query",
        #     "text_decoder.bert.encoder.layer.1.attention.self.value",
        #     "text_decoder.bert.encoder.layer.2.attention.self.query",
        #     "text_decoder.bert.encoder.layer.2.attention.self.value",
        # ]  # Selecting only a few early layers reduces training cost and speeds up adaptation

        ###############################################################################################################
        ## What This Does:
        ##
        ## Vision Encoder (ViT): Modifies deeper layers (6-9), meaning the model will adapt high-level image features instead of early low-level ones.
        ## Text Decoder (BERT): Adjusts mid-to-late attention layers (5-6), which means the model refines text generation logic based on deeper linguistic structures.
        ## Best For:
        ##
        ## More complex fine-tuning tasks where the model needs to deeply understand visual details (e.g., medical images, fine-grained classification).
        ## Situations where the pretrained features are not enough, and the model needs to learn new patterns from scratch.
        ## Downsides:
        ##
        ## Higher compute & memory cost (deeper layers require more adjustments).
        ## Might overwrite pretrained knowledge too aggressively, leading to catastrophic forgetting (losing generalization ability).
        ###############################################################################################################
        target_modules=[
            # Vision Encoder (ViT) - Deeper layers for better image understanding
            "vision_model.encoder.layers.6.self_attn.qkv",
            "vision_model.encoder.layers.7.self_attn.qkv",
            "vision_model.encoder.layers.8.self_attn.qkv",
            "vision_model.encoder.layers.9.self_attn.qkv",

            # Text Decoder - More layers for improved captioning
            "text_decoder.bert.encoder.layer.5.attention.self.query",
            "text_decoder.bert.encoder.layer.5.attention.self.key",
            "text_decoder.bert.encoder.layer.5.attention.self.value",
            "text_decoder.bert.encoder.layer.6.attention.self.query",
            "text_decoder.bert.encoder.layer.6.attention.self.key",
            "text_decoder.bert.encoder.layer.6.attention.self.value"
        ]

    )





    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()  # Print LoRA trainable params for verification




# =============================
# LOAD DATASET & DATALOADER
# =============================
print("Loading dataset...")
image_dir = "coco_sample"
captions_path = "coco_sample/captions.json"

try:
    dataset = CocoDataset(image_dir, captions_path, processor)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    print(f"Loaded dataset with {len(dataset)} images.")
except FileNotFoundError as e:
    print(str(e))
    exit(1)

# =============================
# DEFINE OPTIMIZER
# =============================
# optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)  # Use AdamW optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-5) # Use AdamW optimizer
gradient_accumulation_steps = 4  # Helps stabilize small-batch training


# =============================
# TRAINING LOOP (1 Epoch for Quick Demo)
# =============================
epochs = 60 # Number of epochs for fine-tuning
freeze_vision_encoder = False # Freeze vision encoder during fine-tuning

print("Starting fine-tuning...")
for epoch in range(epochs):
    print(f"Epoch {epoch + 1} begins...")

    if freeze_vision_encoder:
        print("Freezing vision encoder and fine-tuning text decoder only.")
        model.text_decoder.train()
        for param in model.vision_model.parameters():
            param.requires_grad = False  # Properly freeze vision encoder
    else:
        print("Fine-tuning both vision encoder and text decoder.")
        model.train()


    batch_num = 0
    total_loss = 0  # Track total loss

    for batch in dataloader:
        batch_num += 1
        batch = {k: v.to(device) for k, v in batch.items()}  # Move data to GPU

        # Forward pass
        outputs = model(**batch, labels=batch["input_ids"])
        loss = outputs.loss
        total_loss += loss.item()

        # Backward pass (optimize LoRA layers)
        loss.backward()
        # optimizer.step()
        # optimizer.zero_grad()
        if batch_num % gradient_accumulation_steps == 0 or batch_num == len(dataloader) - 1:
            optimizer.step()
            optimizer.zero_grad()  # Reset gradients



        # Print progress every 10 batches
        if batch_num % 10 == 0 or batch_num == len(dataloader):
            avg_loss = total_loss / batch_num
            print(f"Batch {batch_num}/{len(dataloader)} - Loss: {avg_loss:.4f}")

    print(f"Epoch {epoch + 1} completed. Final Loss: {total_loss / len(dataloader):.4f}")

# =============================
# SAVE FINE-TUNED MODEL & PROCESSOR
# =============================
print("Saving fine-tuned model & processor...")
save_path = "blip_coco_finetuned"

model.save_pretrained(save_path)  # Save model weights
processor.save_pretrained(save_path)  # Save processor (tokenizer & config)

print("Fine-tuning complete! Model and processor saved.")


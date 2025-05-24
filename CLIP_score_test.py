from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import torch
import os
from tqdm import tqdm

# Load model and processor
device = "cuda" if torch.cuda.is_available() else "cpu"
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# Image folder and prompt
image_folder = "results/grid_search_25"
prompt = "A high-resolution scenic mountain banner, majestic snow-capped peaks under a clear blue sky, lush green forest at the base, crisp atmosphere."

def compute_clip_score(image_path, text):
    image = Image.open(image_path).convert("RGB")
    inputs = processor(text=[text], images=image, return_tensors="pt", padding=True).to(device)
    outputs = model(**inputs)
    logits_per_image = outputs.logits_per_image
    score = logits_per_image.item()  # Cosine similarity score
    return score

# Evaluate all images in folder
scores = []
for fname in tqdm(os.listdir(image_folder)):
    if fname.lower().endswith((".png", ".jpg", ".jpeg")):
        path = os.path.join(image_folder, fname)
        score = compute_clip_score(path, prompt)
        scores.append((fname, score))

# Sort by score and print
scores.sort(key=lambda x: x[1], reverse=True)
print("Top-scoring images:")
for name, s in scores[:10]:
    print(f"{name}: {s:.4f}")
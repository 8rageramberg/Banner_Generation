from diffusers import AutoPipelineForText2Image
import torch
from PIL import Image
import os
import gc

# === Setup ===
model_id = "runwayml/stable-diffusion-v1-5"
ip_adapter_path = "h94/IP-Adapter"
ip_weights = "ip-adapter_sd15.bin"
subfolder = "models"

prompt = "A wide summer forest banner, trees and sunlight, with a small logo in the bottom-right, vivid colors, cinematic lighting"
negative_prompt = "ugly, blurry, deformed, low quality, text, watermark"
adapter_scale = 0.6
steps = 50

device = "mps"
dtype = torch.float32

# === Logo image for IP-Adapter ===
logo_path = "nike.png"
logo = Image.open(logo_path).convert("RGB").resize((224, 224))  # expected input for IP-Adapter

# === Load model ===
pipe = AutoPipelineForText2Image.from_pretrained(
    model_id,
    torch_dtype=dtype
).to(device)

# === Load IP-Adapter ===
pipe.load_ip_adapter(ip_adapter_path, subfolder=subfolder, weight_name=ip_weights)
pipe.set_ip_adapter_scale(adapter_scale)

# 🛑 Disable NSFW checker properly
pipe.safety_checker = lambda images, **kwargs: (images, [False] * len(images))

# === Generate image ===
generator = torch.Generator(device=device).manual_seed(42)
result = pipe(
    prompt=prompt,
    ip_adapter_image=logo,
    negative_prompt=negative_prompt,
    num_inference_steps=steps,
    guidance_scale=7.5,
    height=384,
    width=768,
    generator=generator
).images[0]

# === Save output ===
output_path = "forest_with_logo.png"
result.save(output_path)
print(f"✅ Saved to {output_path}")

# === Clean up ===

import gc
del pipe, result, generator
gc.collect()
torch.mps.empty_cache()
print("🧽 Memory cleared.")
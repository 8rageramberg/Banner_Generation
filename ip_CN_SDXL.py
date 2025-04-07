from diffusers import AutoPipelineForText2Image, ControlNetModel, StableDiffusionControlNetPipeline, StableDiffusionXLControlNetPipeline
from diffusers.utils import load_image
import torch
from PIL import Image, ImageDraw, ImageFilter
import os
import gc
import cv2
import numpy as np


# === Setup ===
model_id = "stabilityai/stable-diffusion-xl-base-1.0"
ip_adapter_path = "h94/IP-Adapter"
ip_weights = "ip-adapter_sdxl.bin"
subfolder = "sdxl_models"
controlnet_model_id = "diffusers/controlnet-canny-sdxl-1.0" 

prompt = "A photo realistic summer forest banner, sunlight through trees, vivid colors. Include small orange rounded logo in the bottom-right corner"

 
negative_prompt = "ugly, blurry, deformed, low quality, watermark"

seed = 42
steps = 10
height = 384
width = 768
guidance_scale = 7.5
adapter_scale = 0.3 # -> starting to get decent
controlnet_conditioning_scale = 0.7 # -> starting to get decent


try_mps = True
device = "mps" if try_mps and torch.backends.mps.is_available() else "cpu"
dtype = torch.float16  # safer with controlnet on MPS

# === Load logo ===
logo_path = "logos/able.jpg"
logo = Image.open(logo_path).convert("RGB").resize((224, 224))



# === 1. Canny edge map for ControlNet Canny
canvas = np.zeros((height, width), dtype=np.uint8)

# Resize logo smaller (e.g. 128x128)
# === Resize logo smaller (e.g. 128x128 or 160x160)
logo_small = logo.resize((128, 128))
logo_np = np.array(logo_small)
logo_gray = cv2.cvtColor(logo_np, cv2.COLOR_RGB2GRAY)
logo_canny = cv2.Canny(logo_gray, 100, 200)

# Paste Canny edges into corner
canvas = np.zeros((height, width), dtype=np.uint8)
y_offset = height - 128 - 30
x_offset = width - 128 - 30
canvas[y_offset:y_offset+128, x_offset:x_offset+128] = logo_canny

control_image = Image.fromarray(canvas).convert("RGB")


# === Load ControlNet ===
controlnet = ControlNetModel.from_pretrained(
    controlnet_model_id,
    torch_dtype=dtype
).to(device)

# === Load pipeline with ControlNet ===
pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
    model_id,
    controlnet=controlnet,
    torch_dtype=dtype
).to(device)

# === Load IP-Adapter ===
pipe.load_ip_adapter(ip_adapter_path, subfolder=subfolder, weight_name=ip_weights)
pipe.set_ip_adapter_scale(adapter_scale)

# 🛑 Disable NSFW checker
pipe.safety_checker = lambda images, **kwargs: (images, [False] * len(images))

# === Generate image ===
generator = torch.Generator(device=device).manual_seed(seed)
result = pipe(
    prompt=prompt,
    ip_adapter_image=logo,
    image=control_image,
    negative_prompt=negative_prompt,
    num_inference_steps=steps,
    guidance_scale=guidance_scale,
    height=height,
    width=width,
    generator=generator,
    controlnet_conditioning_scale=controlnet_conditioning_scale

).images[0]

# === Save output ===
output_dir = "results/ip_and_control"
os.makedirs(output_dir, exist_ok=True)

base_filename = "forest_with_logo_controlnet_SDXL"
output_path = os.path.join(output_dir, base_filename + ".png")

# Check for duplicates and increment filename
count = 1
while os.path.exists(output_path):
    output_path = os.path.join(output_dir, f"{base_filename}_{count}.png")
    count += 1

result.save(output_path)
print(f"✅ Saved to {output_path}")

# === Clean up ===
del pipe, result, generator, controlnet
gc.collect()
torch.mps.empty_cache()
print("🧽 Memory cleared.")
from diffusers import AutoPipelineForText2Image, ControlNetModel, StableDiffusionControlNetPipeline
from diffusers.utils import load_image
import torch
from PIL import Image, ImageDraw, ImageFilter
import os
import gc
import cv2
import numpy as np


# === Setup ===
model_id = "runwayml/stable-diffusion-v1-5"
ip_adapter_path = "h94/IP-Adapter"
ip_weights = "ip-adapter_sd15.bin"
subfolder = "models"
controlnet_model_id = "lllyasviel/sd-controlnet-canny"  # You can try other types too
# controlnet_model_id = "lllyasviel/sd-controlnet-scribble"   # scribble for spatiel mask


prompt = "A cinematic summer forest banner, sunlight through trees, vivid colors, with a small logo in the bottom-right"

prompt = "A summer forest banner, sunlight through trees, vivid colors. Include small orange rounded logo in the bottom-right"

negative_prompt = "ugly, blurry, deformed, low quality, watermark"
adapter_scale = 0.1
steps = 20
height = 384
width = 768


device = "mps"
dtype = torch.float32  # safer with controlnet on MPS

# === Load logo ===
logo_path = "able.jpg"
logo = Image.open(logo_path).convert("RGB").resize((224, 224))








# === 1. Canny edge map for ControlNet Canny

canvas = np.zeros((height, width), dtype=np.uint8)

# Resize logo smaller (e.g. 128x128)
# === Resize logo smaller (e.g. 128x128 or 160x160)
logo_small = logo.resize((128, 128))

# === Canny edge of logo
logo_np = np.array(logo_small)
logo_gray = cv2.cvtColor(logo_np, cv2.COLOR_RGB2GRAY)
logo_canny = cv2.Canny(logo_gray, 100, 200) # lower thresholds = more detail, higher = cleanr but simpler shapes. (100, 200) (50, 150) (30, 120)

# === Create blank canvas and paste canny in bottom-right
canvas = np.zeros((height, width), dtype=np.uint8)
y_offset = height - logo_canny.shape[0] - 30
x_offset = width - logo_canny.shape[1] - 30
canvas[y_offset:y_offset + logo_canny.shape[0], x_offset:x_offset + logo_canny.shape[1]] = logo_canny

# === Save result
canny_control_image = Image.fromarray(canvas).convert("RGB")
canny_control_image.save("canny_corner_control_image_hd.png")



# === 2. controlNet Scribble
# controlnet_model_id = "lllyasviel/sd-controlnet-scribble"

# # === Convert to white-on-black logo only (mask-like scribble)
# logo_gray = logo.convert("L")  # grayscale
# logo_bw = logo_gray.point(lambda x: 255 if x > 50 else 0).convert("1")  # threshold to get solid white logo

# # === Resize and paste in corner
# logo_scribble = logo_bw.resize((96, 96)).convert("RGB")
# scribble_canvas = Image.new("RGB", (width, height), (0, 0, 0))
# scribble_canvas.paste(logo_scribble, (width - 116, height - 116))

# scribble_canvas.save("scribble_control_image.png")




# === 2. Simple spatial mask with a white dot in bottom-right
# mask = Image.new("RGB", (width, height), (10, 10, 10)) # (0,0,0) -> (10,10,10) Works identically visually, but is tensor-safe âœ…

# dot_size = 128

# draw = ImageDraw.Draw(mask)
# draw.rectangle(
#     (width - dot_size - 20, height - dot_size - 20, width - 20, height - 20),
#     fill=(255, 255, 255)
# )
# mask = mask.filter(ImageFilter.GaussianBlur(4))
# mask.save("logo_pos_mask.png")




# === Use logo as structure image (e.g. a sketch/mask or canny edge input) ===
#control_image = logo.resize((width, height))



# control_image = Image.open("scribble_logo_mask.png")  # <- this were scrible but looked terrible 
#control_image  = Image.open("canny_corner_control_image.png")




# === 4. back to canny: 
# logo_rgba = Image.open("nike.png").convert("RGBA")
# logo_small = logo_rgba.resize((96, 96))

# # Create canvas
# control_image = Image.new("RGBA", (width, height), (0, 0, 0, 0))

# # Paste using alpha mask to preserve cutout shape
# control_image.paste(logo_small, (width - 116, height - 116), mask=logo_small)

# # Flatten to RGB after paste
# control_image = control_image.convert("RGB")
# control_image.save("canny_logo_hint.png")

# img_np = np.array(control_image)
# gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
# edges = cv2.Canny(gray, 100, 200)
# edges_img = Image.fromarray(edges).convert("RGB")
# edges_img.save("canny_logo_hint_edges.png")





# === Load ControlNet ===
controlnet = ControlNetModel.from_pretrained(
    controlnet_model_id,
    torch_dtype=dtype
).to(device)

# === Load pipeline with ControlNet ===
pipe = StableDiffusionControlNetPipeline.from_pretrained(
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
generator = torch.Generator(device=device).manual_seed(42)
result = pipe(
    prompt=prompt,
    ip_adapter_image=logo,
    image=Image.open("canny_corner_control_image_hd.png"),
    negative_prompt=negative_prompt,
    num_inference_steps=steps,
    guidance_scale=7.5,
    height=height,
    width=width,
    generator=generator
).images[0]

# === Save ===
# === Save output ===
output_dir = "results/ip_and_control"
os.makedirs(output_dir, exist_ok=True)

base_filename = "forest_with_logo_controlnet"
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
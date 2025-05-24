import itertools
import os
import gc
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch
from diffusers import StableDiffusionXLControlNetPipeline, ControlNetModel
import matplotlib.pyplot as plt

import time

time_start = time.time()  # Record start time


# === Basic Configs ===
model_id = "stabilityai/stable-diffusion-xl-base-1.0"
ip_adapter_path = "h94/IP-Adapter"
ip_weights = "ip-adapter_sdxl.bin"
subfolder = "sdxl_models"
controlnet_model_id = "diffusers/controlnet-canny-sdxl-1.0" 

prompt = "A photo realistic summer forest banner, sunlight through trees, vivid colors. Include logo in the bottom-right corner"
negative_prompt = "ugly, blurry, deformed, low quality, watermark, "

seed = 42
steps = 100
height = 384
width = 768

# We'll override these in the grid loop:
guidance_scale_default = 7.5  
adapter_scale_default = 0.3
controlnet_conditioning_scale_default = 0.7

grid_guidance = [8.0]         # Guidance scale values
grid_adapter = [0.1, 0.3]           # IP-Adapter scale values
grid_cn = [0.5, 1.0]                # ControlNet conditioning scale values


grid_guidance = [7.0, 10.0]         # Guidance scale values
grid_adapter = [0.1, 0,2]           # IP-Adapter scale values
grid_cn = [0.5, 1.0] 

try_mps = True
device = "mps" if try_mps and torch.backends.mps.is_available() else "cpu"
dtype = torch.float16  # using float16 on MPS

# === Load logo and prepare IP-Adapter image embeds & control image ===
logo_path = "logos/20.jpg"
logo = Image.open(logo_path).convert("RGB").resize((224, 224))
logo.save("resized_logo.png")  # Save to disk

# --- Load pipeline and IP-Adapter (once) ---
controlnet = ControlNetModel.from_pretrained(controlnet_model_id, torch_dtype=dtype).to(device)
pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
    model_id, controlnet=controlnet, torch_dtype=dtype
).to(device)
pipe.load_ip_adapter(ip_adapter_path, subfolder=subfolder, weight_name=ip_weights)
pipe.set_ip_adapter_scale(adapter_scale_default)

# Prepare IP-Adapter image embeds using the helper method (as per Hugging Face docs)
image_embeds = pipe.prepare_ip_adapter_image_embeds(
    ip_adapter_image=logo,
    ip_adapter_image_embeds=None,
    device=device,
    num_images_per_prompt=1,
    do_classifier_free_guidance=True,
)

# --- Generate ControlNet input (Canny edges of the logo placed in the corner) ---
logo_small = logo.resize((128, 128))
logo_np = np.array(logo_small)
logo_gray = cv2.cvtColor(logo_np, cv2.COLOR_RGB2GRAY)
logo_canny = cv2.Canny(logo_gray, 100, 200)
canvas = np.zeros((height, width), dtype=np.uint8)
y_offset = height - 128 - 30
x_offset = width - 128 - 30
canvas[y_offset:y_offset+128, x_offset:x_offset+128] = logo_canny
control_image = Image.fromarray(canvas).convert("RGB")
control_image.save("debug_canny_input.png")

# Disable NSFW checker for testing
pipe.safety_checker = lambda images, **kwargs: (images, [False] * len(images))

# === Grid Search Setup ===

base_dir = "results/grid_search"
grid_dir = base_dir
i = 1
while os.path.exists(grid_dir):
    grid_dir = f"{base_dir}_{i}"
    i += 1
os.makedirs(grid_dir)
print(f"Results will be saved in: {grid_dir}")

# List to store info for collage
results_info = []

# Loop over all combinations
for g, a, cn in itertools.product(grid_guidance, grid_adapter, grid_cn):
    print(f"Generating for Guidance={g}, Adapter={a}, CN={cn}...")
    
    # Update the pipeline parameters
    pipe.set_ip_adapter_scale(a)
    # Set guidance scale and controlnet conditioning scale in the pipeline call
    
    # Reinitialize generator with fixed seed for reproducibility
    generator = torch.Generator(device=device).manual_seed(seed)
    
    result_img = pipe(
        prompt=prompt,
        ip_adapter_image_embeds=image_embeds,  # using prepared embeddings
        image=control_image,
        negative_prompt=negative_prompt,
        num_inference_steps=steps,
        guidance_scale=g,
        controlnet_conditioning_scale=cn,
        height=height,
        width=width,
        generator=generator
    ).images[0]
    
    # Save the individual image with parameter info in the filename
    filename = f"result_g{g}_a{a}_cn{cn}_ss{steps}.png"
    filepath = os.path.join(grid_dir, filename)
    result_img.save(filepath)
    print(f"Saved {filepath}")
    
    results_info.append((result_img, g, a, cn, filepath))
    del result_img, generator
    gc.collect()
    torch.mps.empty_cache()



# --- Matplotlib Collage Generation ---    first try
# Define grid: rows = len(grid_guidance), columns = len(grid_adapter)*len(grid_cn)
combo_list = sorted([(a, cn) for a in grid_adapter for cn in grid_cn])
num_rows = len(grid_guidance)
num_cols = len(combo_list)

fig, axs = plt.subplots(num_rows, num_cols, figsize=(num_cols * 3, num_rows * 3))
fig.subplots_adjust(hspace=0.4, wspace=0.2)

# Ensure axs is a 2D list for consistent indexing
if num_rows == 1:
    axs = [axs]
if num_cols == 1:
    axs = [[ax] for ax in axs]

combo_to_col = {combo: idx for idx, combo in enumerate(combo_list)}
results_info_sorted = sorted(results_info, key=lambda x: (x[1], x[2], x[3]))

# Use a default font for labels
from PIL import ImageFont
font = ImageFont.load_default()

for img, g, a, cn, path in results_info_sorted:
    row = grid_guidance.index(g)
    col = combo_to_col[(a, cn)]
    x = col * width
    y = row * height
    # In the Matplotlib subplot grid:
    ax = axs[row][col]
    ax.imshow(np.array(img))
    ax.axis("off")
    title_text = f"G:{g}, A:{a}, CN:{cn}"
    ax.set_title(title_text, fontsize=10, pad=4, backgroundcolor='white')

collage_path_mp = os.path.join(grid_dir, "collage_matplotlib.png")
plt.savefig(collage_path_mp, bbox_inches="tight", dpi=300)
print(f"Matplotlib Collage saved to {collage_path_mp}")



# second try matplooooooooot
results_dict = {}
for item in results_info_sorted:
    img, g, a, cn, path = item
    results_dict[(g, a, cn)] = img

# Create a sorted list of adapter/controlnet combinations for columns.
combo_list = sorted([(a, cn) for a in grid_adapter for cn in grid_cn])
num_rows = len(grid_guidance)
num_cols = len(combo_list)

# Create a Matplotlib figure with subplots arranged in a grid.
fig, axs = plt.subplots(num_rows, num_cols, figsize=(num_cols * 3, num_rows * 3))
fig.subplots_adjust(hspace=0.4, wspace=0.2)

# Ensure axs is always a 2D list for consistent indexing.
if num_rows == 1:
    axs = [axs]
if num_cols == 1:
    axs = [[ax] for ax in axs]

# Loop over each guidance value (row) and each combination (column).
for i, g in enumerate(grid_guidance):
    for j, (a, cn) in enumerate(combo_list):
        ax = axs[i][j]
        key = (g, a, cn)
        if key in results_dict:
            # Display the generated image.
            img = results_dict[key]
            ax.imshow(np.array(img))
        else:
            # If no image was produced for this combination, fill with white and label "Empty".
            blank = np.ones((height, width, 3), dtype=np.uint8) * 255
            ax.imshow(blank)
            ax.text(0.5, 0.5, "Empty", ha='center', va='center', fontsize=12, color='black', transform=ax.transAxes)
        ax.axis("off")
        # Set a title with a white background for clarity.
        title_text = f"G:{g}\nA:{a}, CN:{cn}"
        ax.set_title(title_text, fontsize=10, pad=4, backgroundcolor='white')

# Save the collage image.
collage_path_mp = os.path.join(grid_dir, "collage_matplotlib_2.png")
plt.savefig(collage_path_mp, bbox_inches="tight", dpi=300)
print(f"Matplotlib Collage saved to {collage_path_mp}")



time_end = time.time()  # ⏱ Record end time
elapsed = time_end - time_start

# Convert to minutes
elapsed_minutes = elapsed / 60

print(f"Script finished in {elapsed:.2f} seconds ({elapsed_minutes:.2f} minutes)")


from bot import send_telegram_message
bot_token = ""
chat_id = ""
send_telegram_message(
    f"Your Grid Search has finished! Elapsed time: {elapsed_minutes:.2f} minutes",
    bot_token,
    chat_id
)

# === Clean up ===
del pipe, controlnet, image_embeds
gc.collect()
torch.mps.empty_cache()
print("🧽 Memory cleared.")
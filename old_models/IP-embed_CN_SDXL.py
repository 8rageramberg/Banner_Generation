from diffusers import StableDiffusionXLControlNetPipeline, ControlNetModel
from diffusers.utils import load_image
import torch
from PIL import Image
import os
import numpy as np
import cv2
import gc


# === Configs ===
model_id = "stabilityai/stable-diffusion-xl-base-1.0"
ip_adapter_path = "h94/IP-Adapter"
ip_weights = "ip-adapter_sdxl.bin"
subfolder = "sdxl_models"
controlnet_model_id = "diffusers/controlnet-canny-sdxl-1.0" 

prompt = "A photo realistic summer forest banner, sunlight through trees, vivid colors. Include small orange rounded logo in the bottom-right corner"
negative_prompt = "ugly, blurry, deformed, low quality, watermark"

seed = 42
steps = 20
height = 384
width = 768
guidance_scale = 7.5
adapter_scale = 0.3
controlnet_conditioning_scale = 0.7

try_mps = True
device = "mps" if try_mps and torch.backends.mps.is_available() else "cpu"
dtype = torch.float16  # safer with controlnet on MPS

# === Load logo ===
logo_path = "logos/able.jpg"
logo = Image.open(logo_path).convert("RGB").resize((224, 224))

# --- Get IP-Adapter image embeddings using the helper method ---
# Here we follow the Hugging Face docs exactly.
# Note: You don't need to manually load a CLIP model—use the pipeline's built-in helper.
# First, load the pipeline (without IP-Adapter embedding injection):
controlnet = ControlNetModel.from_pretrained(controlnet_model_id, torch_dtype=dtype).to(device)
pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
    model_id, controlnet=controlnet, torch_dtype=dtype
).to(device)
pipe.load_ip_adapter(ip_adapter_path, subfolder=subfolder, weight_name=ip_weights)
pipe.set_ip_adapter_scale(adapter_scale)

# Prepare the IP-Adapter image embeds using the provided helper method:
image_embeds = pipe.prepare_ip_adapter_image_embeds(
    ip_adapter_image=logo,
    ip_adapter_image_embeds=None,
    device=device,
    num_images_per_prompt=1,
    do_classifier_free_guidance=True,
)

# === Generate ControlNet input (Canny of logo in corner) ===
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

# === Generate image ===
generator = torch.Generator(device=device).manual_seed(seed)
result = pipe(
    prompt=prompt,
    ip_adapter_image_embeds=image_embeds,  # pass the prepared embeddings
    image=control_image,
    negative_prompt=negative_prompt,
    num_inference_steps=steps,
    guidance_scale=guidance_scale,
    controlnet_conditioning_scale=controlnet_conditioning_scale,
    height=height,
    width=width,
    generator=generator
).images[0]

# === Save output ===
output_dir = "results/ip_adapter_image_embeds_SDXL"
os.makedirs(output_dir, exist_ok=True)
filename = "forest_with_logo_embed.png"
i = 1
while os.path.exists(os.path.join(output_dir, filename)):
    filename = f"forest_with_logo_embed_{i}.png"
    i += 1
output_path = os.path.join(output_dir, filename)
result.save(output_path)
print(f"✅ Saved to: {output_path}")

# === Clean up ===
del pipe, result, generator, controlnet, image_embeds
gc.collect()
torch.mps.empty_cache()
print("🧽 Memory cleared.")


import requests
def send_telegram_message(message, bot_token, chat_id):
    url = f""
    payload = {"chat_id": chat_id, "text": message}
    try:
        response = requests.post(url, json=payload)
        if response.status_code == 200:
            print("Telegram notification sent!")
        else:
            print("Failed to send Telegram notification:", response.text)
    except Exception as e:
        print("Error sending Telegram notification:", e)

# Replace with your actual bot token and chat id
bot_token = ""
chat_id = ""
send_telegram_message("Your script has finished!", bot_token, chat_id)


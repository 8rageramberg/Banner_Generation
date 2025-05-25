"""
This script performs grid-based image generation using Stable Diffusion XL with ControlNet and IP-Adapter.
It generates images with a logo placed in the bottom-right.
"""

import os
import gc
import time
import cv2
import numpy as np
from PIL import Image
import torch
from diffusers import StableDiffusionXLControlNetPipeline, ControlNetModel
import matplotlib.pyplot as plt
from diffusers.image_processor import IPAdapterMaskProcessor
from bot import send_telegram_message  # Make sure this module is in your PYTHONPATH


# Selects the appropriate compute device (CUDA if available, otherwise CPU).
# CUDA with at <16GB GPU memory (e.g., A100, V100, RTX 3090) and high host RAM (≥32GB) is highly recommended,
# as Stable Diffusion XL pipelines and IP-Adapter processing are memory-intensive.
def get_device():
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"✅ Using CUDA: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("⚠️ Using CPU")
    return device

# Returns optimal tensor data type based on the compute device
def get_dtype(device):
    return torch.float16 if device.type == "cuda" else torch.float32

# Places a smaller image (patch) into the bottom-right corner of a larger base image
def place_patch_bottom_right(base, patch, margin=30):
    h, w = patch.shape[:2]
    H, W = base.shape[:2]
    x_off = W - w - margin
    y_off = H - h - margin
    base[y_off:y_off + h, x_off:x_off + w] = patch
    return base

# Loads the logo and prepares two output images:
# - control image using Canny edges
# - alpha mask for IP-Adapter masking
def prepare_control_and_mask(logo_path, height, width, size=(224, 224), margin=30):
    logo = Image.open(logo_path).convert("RGBA").resize(size)
    logo_np = np.array(logo)
    gray = cv2.cvtColor(logo_np[:, :, :3], cv2.COLOR_RGB2GRAY)
    canny = cv2.Canny(gray, 100, 200)
    control_canvas = np.zeros((height, width), dtype=np.uint8)
    control_canvas = place_patch_bottom_right(control_canvas, canny, margin)
    alpha = logo_np[:, :, 3]
    mask_canvas = np.zeros((height, width), dtype=np.uint8)
    mask_canvas = place_patch_bottom_right(mask_canvas, alpha, margin)
    return Image.fromarray(control_canvas).convert("RGB"), Image.fromarray(mask_canvas)

# Initializes and configures the Stable Diffusion XL pipeline with ControlNet and IP-Adapter
def setup_pipeline(model_id, ip_adapter_path, ip_weights, subfolder, controlnet_model_id, device, dtype, adapter_scale_default):
    controlnet = ControlNetModel.from_pretrained(controlnet_model_id, torch_dtype=dtype).to(device)
    pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
        model_id, controlnet=controlnet, torch_dtype=dtype
    ).to(device)
    pipe.load_ip_adapter(ip_adapter_path, subfolder=subfolder, weight_name=ip_weights)
    pipe.set_ip_adapter_scale(adapter_scale_default)
    # By default, the safety checker is enabled in Stable Diffusion XL.
    # To disable NSFW filtering (which may suppress outputs), uncomment the line below.
    # pipe.safety_checker = lambda images, **kwargs: (images, [False] * len(images))
    return pipe, controlnet

# Converts the logo image into IP-Adapter-compatible image embeddings
def prepare_image_embeds(pipe, logo, device):
    return pipe.prepare_ip_adapter_image_embeds(
        ip_adapter_image=logo,
        device=device,
        num_images_per_prompt=1,
        do_classifier_free_guidance=True,
    )

# Creates a callback to disable the adapter after a defined fraction of inference steps
def hard_cutoff_adapter_callback_creator(adapter_scale_default, steps, cutoff_ratio):
    cutoff_step = int(steps * cutoff_ratio)
    def callback(pipe_obj, step: int, timestep: float, callback_kwargs, **kwargs):
        scale = adapter_scale_default if step < cutoff_step else 0.0
        pipe_obj.set_ip_adapter_scale(scale)
        return {}
    return callback

# (Unused) Creates a callback to activate the adapter only after a certain number of steps
def late_activation_callback_creator(adapter_scale_late, steps, activate_ratio):
    activate_step = int(steps * activate_ratio)
    def callback(pipe_obj, step: int, timestep: float, callback_kwargs, **kwargs):
        scale = adapter_scale_late if step >= activate_step else 0.0
        pipe_obj.set_ip_adapter_scale(scale)
        return {}
    return callback

# Performs a small grid search over adapter scales, guidance, and cutoff points
# Saves output images for each parameter combination
def run_small_grid(pipe, prompt, negative_prompt, image_embeds, control_image, ip_adapter_masks, grid_dir, seeds, device):
    # Grid parameters
    grid_adapter = [0.1, 0.2]
    grid_cn = [0.7]
    grid_guidance = [6.0]
    grid_cutoffs = [1/10, 1/3]
    steps = 50
    width, height = 2048, 1024
    results = []


    # Grid loop with high readability and storage of results
    for seed in seeds:
        for g in grid_guidance:
            for a in grid_adapter:
                for cn in grid_cn:
                    for co in grid_cutoffs:
                        pipe.set_ip_adapter_scale(a)
                        callback = hard_cutoff_adapter_callback_creator(a, steps, co)
                        gen = torch.Generator(device=device).manual_seed(seed)
                        output = pipe(prompt=prompt, negative_prompt=negative_prompt,
                                      ip_adapter_image_embeds=image_embeds,
                                      ip_adapter_masks=ip_adapter_masks,
                                      image=control_image, num_inference_steps=steps,
                                      guidance_scale=g, controlnet_conditioning_scale=cn,
                                      height=height, width=width, generator=gen,
                                      callback_on_step_end=callback, callback_steps=1)

                        img = output.images[0]
                        fname = f"grid_seed{seed}_g{g}_a{a}_cn{cn}_co{int(co*100)}.png"
                        fpath = os.path.join(grid_dir, fname)
                        img.save(fpath)
                        results.append(img)
                        del img, gen
                        gc.collect()
    return results

# Creates and saves a side-by-side collage of generated images
def generate_collage(images, collage_path):
    cols = len(images)
    fig, axs = plt.subplots(1, cols, figsize=(cols * 6, 6))
    for i, img in enumerate(images):
        axs[i].imshow(np.array(img))
        axs[i].axis("off")
    plt.tight_layout()
    plt.savefig(collage_path, dpi=300)
    plt.close()

def main():
    time_start = time.time()
    # Define generation scenarios with associated logos and descriptions
    prompts = [
        {"prompt": "A majestic mountain banner, snow-covered peaks, vibrant sky, include logo in bottom-right corner.",
         "logo_path": "logos/foxpng.png",
         "name": "mountain"},

        {"prompt": "A futuristic cityscape banner at night, neon lights, reflections, include logo bottom-right.",
         "logo_path": "logos/logoipsum.png",
         "name": "city"},

        {"prompt": "A cozy autumn forest banner, golden leaves and sunlight, include logo in bottom-right corner.",
         "logo_path": "logos/rabbit.png",
         "name": "forest"},
    ]

    # Set of qualities to suppress during image generation
    negative_prompt = (
    "blurry, distorted, low quality, watermark, low resolution, noisy, artifacts, "
    "oversaturated, overexposed, poorly drawn, deformed, unrealistic, bad anatomy, "
    "extra limbs, missing limbs, mutated, text, signature, frame, border, cropped, "
    "aliasing, compression artifacts"
    )
    seeds = [42, 77, 123]

    # Model and pipeline configuration
    model_id = "stabilityai/stable-diffusion-xl-base-1.0"
    ip_adapter_path = "h94/IP-Adapter"
    ip_weights = "ip-adapter_sdxl.bin"
    subfolder = "sdxl_models"
    controlnet_model_id = "diffusers/controlnet-canny-sdxl-1.0"

    device = get_device()
    dtype = get_dtype(device)

    # Run generation for each configured prompt
    for config in prompts:
        grid_dir = f"results/final_{config['name']}"
        os.makedirs(grid_dir, exist_ok=True)

        # Prepare Canny control image and mask for the logo
        control_image, mask = prepare_control_and_mask(config["logo_path"], 1024, 2048)
        processor = IPAdapterMaskProcessor()
        ip_adapter_masks = processor.preprocess([mask], height=1024, width=2048)

        # Load and configure the diffusion pipeline
        pipe, controlnet = setup_pipeline(model_id, ip_adapter_path, ip_weights, subfolder,
                                          controlnet_model_id, device, dtype, 0.1)
        # if device.type == "cuda":
        #     pipe.enable_xformers_memory_efficient_attention()

        # Convert logo image into embedding format
        logo_highres = Image.open(config["logo_path"]).convert("RGB").resize((768, 768))
        image_embeds = prepare_image_embeds(pipe, logo_highres, device)

        # Run inference across parameter grid
        images = run_small_grid(pipe, config["prompt"], negative_prompt, image_embeds,
                                control_image, ip_adapter_masks, grid_dir, seeds, device)

        # Save a collage of output images for each prompt
        collage_path = os.path.join(grid_dir, f"collage_{config['name']}.png")
        generate_collage(images, collage_path)

        # Clean up memory before next run
        del pipe, controlnet, image_embeds
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()


            # === Wrap up ===
    time_end = time.time()
    elapsed = time_end - time_start
    elapsed_minutes = elapsed / 60
    print(f"Script finished in {elapsed:.2f} seconds ({elapsed_minutes:.2f} minutes)")

    # Send a Telegram message
    try:
        bot_token = ""
        chat_id = ""
        if bot_token and chat_id:
            send_telegram_message(
                f"Your Grid Search has finished! Elapsed time: {elapsed_minutes:.2f} minutes",
                bot_token,
                chat_id
            )
        else:
            print("ℹ️ Telegram bot token or chat ID not set; skipping notification.")
    except Exception as e:
        print(f"⚠️ Telegram notification failed: {e}")

if __name__ == "__main__":
    main()
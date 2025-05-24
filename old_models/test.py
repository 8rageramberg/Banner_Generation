import os
import gc
import itertools
import time
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch
from diffusers import StableDiffusionXLControlNetPipeline, ControlNetModel
import matplotlib.pyplot as plt
from botTest import send_telegram_message  # Make sure this module is in your PYTHONPATH

from diffusers.image_processor import IPAdapterMaskProcessor


def get_device():
    """Determines the computing device; prefers CUDA on cluster."""
    return "cuda" if torch.cuda.is_available() else "mps"

def load_logo(logo_path, size=(224, 224)):
    """Loads and resizes the logo image; saves a resized copy."""
    logo = Image.open(logo_path).convert("RGB").resize(size)
    logo.save("resized_logo.png")
    return logo


def prepare_control_image(logo, height, width):
    """
    Creates a control image using Canny edges extracted from a smaller version of the logo.
    The result is placed in a canvas at the bottom-right corner.
    """
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
    return control_image


def setup_pipeline(model_id, ip_adapter_path, ip_weights, subfolder, controlnet_model_id, device, dtype, adapter_scale_default):
    """
    Loads the ControlNet and Stable Diffusion XL pipeline,
    attaches the IP-Adapter, and disables the NSFW checker.
    
    **UPDATE:** Here, adapter_scale_default is used for initial setup;
    during grid search, it will be overridden by each grid value.
    """
    controlnet = ControlNetModel.from_pretrained(controlnet_model_id, torch_dtype=dtype).to(device)
    pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
        model_id, controlnet=controlnet, torch_dtype=dtype
    ).to(device)
    pipe.load_ip_adapter(ip_adapter_path, subfolder=subfolder, weight_name=ip_weights)
    # Set initial adapter scale (will be updated later)
    pipe.set_ip_adapter_scale(adapter_scale_default)
    # Disable NSFW checker for testing
    pipe.safety_checker = lambda images, **kwargs: (images, [False] * len(images))
    return pipe, controlnet


def prepare_image_embeds(pipe, logo, device):
    """
    Prepares the image embeddings for the IP-Adapter.
    """
    return pipe.prepare_ip_adapter_image_embeds(
        ip_adapter_image=logo,
        ip_adapter_image_embeds=None,
        device=device,
        num_images_per_prompt=1,
        do_classifier_free_guidance=True,
    )


def get_grid_directory(base_dir):
    """
    Creates a unique grid results directory (if the base exists, adds a suffix).
    """
    grid_dir = base_dir
    i = 1
    while os.path.exists(grid_dir):
        grid_dir = f"{base_dir}_{i}"
        i += 1
    os.makedirs(grid_dir)
    print(f"Results will be saved in: {grid_dir}")
    return grid_dir

def dynamic_adapter_callback_creator(adapter_scale_default, steps):
    """
    Returns a callback function that updates the IP-adapter scale dynamically.
    The scale decays linearly from the provided starting value to 0 over the number of steps.
    """
    def dynamic_adapter_callback(pipe_obj, step: int, timestep: float, callback_kwargs, **kwargs):
        # Decay: from adapter_scale_default down to 0 linearly
        new_scale = adapter_scale_default * max(0.0, 1 - step / float(steps))
        pipe_obj.set_ip_adapter_scale(new_scale)
        print(f"Step {step}/{steps}: Adapter scale = {new_scale:.3f}")
        # Return an empty dict to satisfy pipeline expectations.
        return {}
    return dynamic_adapter_callback


def dynamic_adapter_callback_creator_cutoff(adapter_scale_default, steps, active_ratio=0.2):
    """
    Returns a callback that applies the IP-adapter only during the first `active_ratio` of steps.
    After that, the adapter scale is set to 0.
    
    Example: active_ratio=0.2 means adapter is active for 20% of steps.
    """
    active_steps = int(steps * active_ratio)

    def dynamic_adapter_callback(pipe_obj, step: int, timestep: float, callback_kwargs, **kwargs):
        new_scale = adapter_scale_default if step < active_steps else 0.0
        pipe_obj.set_ip_adapter_scale(new_scale)
        print(f"Step {step}/{steps}: Adapter scale = {new_scale:.3f}")
        return {}

    return dynamic_adapter_callback


def create_ip_adapter_mask(control_image, height, width, blurred=False, blur_padding=20):
    """
    Creates a binary or soft circular mask based on the control image's active region.
    """
    control_np = np.array(control_image.convert("L"))
    coords = cv2.findNonZero(control_np)
    x, y, w, h = cv2.boundingRect(coords)
    x0, y0, x1, y1 = x, y, x + w, y + h

    if not blurred:
        mask = np.zeros((height, width), dtype=np.uint8)
        cv2.rectangle(mask, (x0, y0), (x1, y1), 255, thickness=-1)
    else:
        cx, cy = (x0 + x1) // 2, (y0 + y1) // 2
        radius = max((x1 - x0), (y1 - y0)) // 2 + blur_padding
        Y, X = np.ogrid[:height, :width]
        dist_from_center = np.sqrt((X - cx)**2 + (Y - cy)**2)
        mask = np.clip(1 - (dist_from_center / radius), 0, 1)
        mask = (mask * 255).astype(np.uint8)

    return Image.fromarray(mask)


def run_grid_search(pipe, prompt, negative_prompt, steps, grid_guidance, grid_adapter, grid_cn,
                    image_embeds, control_image, ip_adapter_masks, seed, device, grid_dir, height, width):
    """
    Iterates over combinations of guidance scale, adapter scale (the starting value), and 
    ControlNet conditioning scale, generating images for each combination and saving them.
    For each grid cell, a dynamic callback is created that decays the adapter scale from the given
    starting value to 0 over the diffusion steps.
    
    Returns a list with results info for collage generation.
    """
    results_info = []
    # Loop over guidance, adapter scale, and ControlNet conditioning scale.
    for g, a, cn in itertools.product(grid_guidance, grid_adapter, grid_cn):
        print(f"Generating for Guidance={g}, Adapter={a}, CN={cn} with dynamic IP-adapter scale...")
        # Reset IP-adapter scale to the starting value for this iteration.
        pipe.set_ip_adapter_scale(a)
        # Create a dynamic callback for the current adapter scale 'a'
        dynamic_callback = dynamic_adapter_callback_creator_cutoff(a, steps, active_ratio=0.4)
        generator = torch.Generator(device=device).manual_seed(seed)
        result = pipe(
            prompt=prompt,
            ip_adapter_image_embeds=image_embeds,
            ip_adapter_masks=ip_adapter_masks,  # 👈 NEW
            image=control_image,
            negative_prompt=negative_prompt,
            num_inference_steps=steps,
            guidance_scale=g,
            controlnet_conditioning_scale=cn,
            height=height,
            width=width,
            generator=generator,
            callback_on_step_end=dynamic_callback,
            callback_steps=1
        )
        result_img = result.images[0]

        # Save the individual image with parameter info in the filename.
        filename = f"result_g{g}_A{a}_CN{cn}_ss{steps}.png"
        filepath = os.path.join(grid_dir, filename)
        result_img.save(filepath)
        print(f"Saved {filepath}")

        results_info.append((result_img, g, a, cn, filepath))
        del result_img, generator
        gc.collect()
        torch.mps.empty_cache()
    return results_info


def generate_collages(results_info, grid_guidance, grid_adapter, grid_cn, width, height, grid_dir):
    """
    Generates two collages using matplotlib: one with a straightforward grid
    and one as a second arrangement. Saves both collages to disk.
    
    The collage titles include guidance, adapter, and ControlNet conditioning values.
    """
    # First collage: grid arranged by guidance and adapter/ControlNet pairs.
    combo_list = sorted([(a, cn) for a in grid_adapter for cn in grid_cn])
    num_rows = len(grid_guidance)
    num_cols = len(combo_list)
    fig, axs = plt.subplots(num_rows, num_cols, figsize=(num_cols * 3, num_rows * 3))
    fig.subplots_adjust(hspace=0.4, wspace=0.2)
    if num_rows == 1:
        axs = [axs]
    if num_cols == 1:
        axs = [[ax] for ax in axs]

    combo_to_col = {combo: idx for idx, combo in enumerate(combo_list)}
    # Sort results_info by guidance, adapter, then CN.
    results_info_sorted = sorted(results_info, key=lambda x: (x[1], x[2], x[3]))
    for img, g, a, cn, path in results_info_sorted:
        row = grid_guidance.index(g)
        col = combo_to_col[(a, cn)]
        ax = axs[row][col]
        ax.imshow(np.array(img))
        ax.axis("off")
        title_text = f"G:{g}, A:{a}, CN:{cn}"
        ax.set_title(title_text, fontsize=10, pad=4, backgroundcolor='white')
    collage_path_mp = os.path.join(grid_dir, "collage_matplotlib.png")
    plt.savefig(collage_path_mp, bbox_inches="tight", dpi=300)
    print(f"Matplotlib Collage saved to {collage_path_mp}")


def cleanup(pipe, controlnet, image_embeds):
    """Clears the main objects from memory and empties the GPU cache."""
    del pipe, controlnet, image_embeds
    gc.collect()
    torch.cuda.empty_cache()
    print("🧽 Memory cleared.")

def main():
    time_start = time.time()

    # === Basic Configurations ===
    model_id = "stabilityai/stable-diffusion-xl-base-1.0"
    ip_adapter_path = "h94/IP-Adapter"
    ip_weights = "ip-adapter_sdxl.bin"
    subfolder = "sdxl_models"
    controlnet_model_id = "diffusers/controlnet-canny-sdxl-1.0"

    prompt = "A photo realistic summer forest banner, sunlight through trees, vivid colors. Include logo in the bottom-right corner"
    negative_prompt = "ugly, blurry, deformed, low quality, watermark, "
    seed = 42
    steps = 1
    height = 384
    width = 768

    # Grid search parameters:
    # **UPDATE:** grid_adapter is now a list of starting adapter scales.
    grid_adapter = [1.0, 1.5]
    grid_guidance = [9.0]
    grid_cn = [0.8]

    # For pipeline setup, use the first adapter value.
    adapter_scale_default = grid_adapter[0]

    try_mps = True
    device = get_device()
    dtype = torch.float16

    # === Load Logo and Prepare Images ===
    logo_path = "logos/19.jpg"
    logo = load_logo(logo_path, size=(224, 224))
    control_image = prepare_control_image(logo, height, width)

    # === IP-Adapter Masking ===
    logo_mask = create_ip_adapter_mask(control_image, height, width, blurred=True)
    processor = IPAdapterMaskProcessor()
    ip_adapter_masks = processor.preprocess([logo_mask], height=height, width=width)


    # === Setup Pipeline ===
    pipe, controlnet = setup_pipeline(model_id, ip_adapter_path, ip_weights, subfolder,
                                      controlnet_model_id, device, dtype, adapter_scale_default)
    image_embeds = prepare_image_embeds(pipe, logo, device)

    # === Grid Search Setup ===
    grid_dir = get_grid_directory("results/grid_search")
    # Note: dynamic_callback will be created inside run_grid_search for each adapter value.


    
    # Run grid search and collect results.
    results_info = run_grid_search(
    pipe, prompt, negative_prompt, steps,
    grid_guidance, grid_adapter, grid_cn,
    image_embeds, control_image, ip_adapter_masks,  
    seed, device, grid_dir, height, width
)

    # Generate collages from the grid search results.
    generate_collages(results_info, grid_guidance, grid_adapter, grid_cn, width, height, grid_dir)

    # === Wrap up ===
    time_end = time.time()
    elapsed = time_end - time_start
    elapsed_minutes = elapsed / 60
    print(f"Script finished in {elapsed:.2f} seconds ({elapsed_minutes:.2f} minutes)")

    # Send a Telegram message
    bot_token = ""
    chat_id = ""
    send_telegram_message(
        f"Your Grid Search has finished! Elapsed time: {elapsed_minutes:.2f} minutes",
        bot_token,
        chat_id
    )

    # Cleanup resources.
    cleanup(pipe, controlnet, image_embeds)

 
if __name__ == "__main__":
    main()

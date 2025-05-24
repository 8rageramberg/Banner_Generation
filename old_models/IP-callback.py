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
from bot import send_telegram_message  # Make sure this module is in your PYTHONPATH


def get_device(try_mps=True):
    """Determines the computing device; uses MPS if available and requested."""
    return "mps" if try_mps and torch.backends.mps.is_available() else "cpu"


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
    attaches the IP-Adapter and disables the NSFW checker.
    """
    controlnet = ControlNetModel.from_pretrained(controlnet_model_id, torch_dtype=dtype).to(device)
    pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
        model_id, controlnet=controlnet, torch_dtype=dtype
    ).to(device)
    pipe.load_ip_adapter(ip_adapter_path, subfolder=subfolder, weight_name=ip_weights)
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
    The scale decays linearly from the default value to 0 over the number of steps.
    """
    def dynamic_adapter_callback(pipe_obj, step: int, timestep: float, callback_kwargs, **kwargs):
        new_scale = adapter_scale_default * max(0.0, 1 - step / float(steps))
        pipe_obj.set_ip_adapter_scale(new_scale)
        print(f"Step {step}/{steps}: Adapter scale = {new_scale:.3f}")
        # Return an empty dict to satisfy pipeline expectations.
        return {}
    return dynamic_adapter_callback


def run_grid_search(pipe, prompt, negative_prompt, steps, grid_guidance, grid_adapter, grid_cn,
                    image_embeds, control_image, seed, device, grid_dir, dynamic_callback, height, width):
    """
    Iterates over combinations of guidance scale, IP-adapter scale, and ControlNet conditioning scale,
    generating images for each combination and saving them.
    Returns a list with results info for collage generation.
    """
    results_info = []
    for g, a, cn in itertools.product(grid_guidance, grid_adapter, grid_cn):
        print(f"Generating for Guidance={g}, Adapter={a}, CN={cn}...")
        # Reset IP-adapter scale to the starting value for this iteration
        pipe.set_ip_adapter_scale(a)
        generator = torch.Generator(device=device).manual_seed(seed)
        result = pipe(
            prompt=prompt,
            ip_adapter_image_embeds=image_embeds,
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

        # Save the individual image with parameter info in the filename
        filename = f"result_g{g}_a{a}_cn{cn}_ss{steps}.png"
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
    """
    # First collage: grid arranged by guidance and adapter/controlnet pairs.
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

    # Second collage: alternative arrangement
    results_dict = {}
    for item in results_info_sorted:
        img, g, a, cn, path = item
        results_dict[(g, a, cn)] = img
    combo_list = sorted([(a, cn) for a in grid_adapter for cn in grid_cn])
    num_rows = len(grid_guidance)
    num_cols = len(combo_list)
    fig, axs = plt.subplots(num_rows, num_cols, figsize=(num_cols * 3, num_rows * 3))
    fig.subplots_adjust(hspace=0.4, wspace=0.2)
    if num_rows == 1:
        axs = [axs]
    if num_cols == 1:
        axs = [[ax] for ax in axs]
    for i, g in enumerate(grid_guidance):
        for j, (a, cn) in enumerate(combo_list):
            ax = axs[i][j]
            key = (g, a, cn)
            if key in results_dict:
                ax.imshow(np.array(results_dict[key]))
            else:
                blank = np.ones((height, width, 3), dtype=np.uint8) * 255
                ax.imshow(blank)
                ax.text(0.5, 0.5, "Empty", ha='center', va='center', fontsize=12,
                        color='black', transform=ax.transAxes)
            ax.axis("off")
            title_text = f"G:{g}\nA:{a}, CN:{cn}"
            ax.set_title(title_text, fontsize=10, pad=4, backgroundcolor='white')
    collage_path_mp2 = os.path.join(grid_dir, "collage_matplotlib_2.png")
    plt.savefig(collage_path_mp2, bbox_inches="tight", dpi=300)
    print(f"Matplotlib Collage saved to {collage_path_mp2}")


def cleanup(pipe, controlnet, image_embeds):
    """Clears the main objects from memory and empties the GPU cache."""
    del pipe, controlnet, image_embeds
    gc.collect()
    torch.mps.empty_cache()
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
    steps = 10
    height = 384
    width = 768
    adapter_scale_default = 0.3

    # Grid search parameters
    grid_guidance = [7.0]
    grid_adapter = [0.1, 0.3]
    grid_cn = [1.0]

    try_mps = True
    device = get_device(try_mps)
    dtype = torch.float16

    # === Load Logo and Prepare Images ===
    logo_path = "logos/20.jpg"
    logo = load_logo(logo_path, size=(224, 224))
    control_image = prepare_control_image(logo, height, width)

    # === Setup Pipeline ===
    pipe, controlnet = setup_pipeline(model_id, ip_adapter_path, ip_weights, subfolder,
                                      controlnet_model_id, device, dtype, adapter_scale_default)
    image_embeds = prepare_image_embeds(pipe, logo, device)

    # === Grid Search Setup ===
    grid_dir = get_grid_directory("results/grid_search")
    dynamic_callback = dynamic_adapter_callback_creator(adapter_scale_default, steps)

    # Run grid search and collect results.
    results_info = run_grid_search(pipe, prompt, negative_prompt, steps, grid_guidance, grid_adapter, grid_cn,
                                   image_embeds, control_image, seed, device, grid_dir, dynamic_callback, height, width)

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
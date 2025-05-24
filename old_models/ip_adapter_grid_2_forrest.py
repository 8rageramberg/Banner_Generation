import os
import platform
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

from diffusers.image_processor import IPAdapterMaskProcessor

#def get_device(try_mps=True):
    #"""Determines the computing device; uses MPS if available and requested."""
    #return "mps" if try_mps and torch.backends.mps.is_available() else "cpu"


def get_device(preferred_gpu=0, verbose=True):
    """
    Detect and return the best available device:
    - CUDA (with GPU selection)
    - MPS (macOS Metal)
    - CPU fallback

    preferred_gpu: Index in CUDA_VISIBLE_DEVICES, not global GPU ID.
    """

    if "CUDA_VISIBLE_DEVICES" in os.environ:
        visible_devices = os.environ["CUDA_VISIBLE_DEVICES"].split(",")
        if preferred_gpu >= len(visible_devices):
            raise ValueError(f"Requested GPU {preferred_gpu}, but only {len(visible_devices)} visible via CUDA_VISIBLE_DEVICES: {visible_devices}")
    else:
        visible_devices = list(range(torch.cuda.device_count()))

    if torch.cuda.is_available():
        device = torch.device(f"cuda:{preferred_gpu}")
        if verbose:
            print(f"✅ CUDA is available. {len(visible_devices)} visible GPUs: {visible_devices}")
            for i in range(torch.cuda.device_count()):
                print(f"  🔹 GPU {i}: {torch.cuda.get_device_name(i)}")
            print(f"🎯 Selected device: cuda:{preferred_gpu}")
    elif platform.system() == "Darwin" and torch.backends.mps.is_available():
        device = torch.device("mps")
        if verbose:
            print("✅ Using Apple MPS backend (macOS)")
    else:
        device = torch.device("cpu")
        if verbose:
            print("⚠️ CUDA and MPS not available. Falling back to CPU.")
    return device

def get_dtype(device):
    """
    Returns torch.float16 for CUDA or MPS devices, torch.float32 otherwise.
    """
    return torch.float16 if device.type in ("cuda", "mps") else torch.float32


def load_logo(logo_path, size=(224, 224)):
    """Loads and resizes the logo image; saves a resized copy."""
    logo = Image.open(logo_path).convert("RGB").resize(size)
    #
    logo.save("resized_logo.png")
    return logo


def prepare_control_image_old(
    logo: Image.Image,
    height: int,
    width: int,
    region_scale: float = 0.1,
    margin_scale: float = 0.02
) -> Image.Image:
    """
    Creates a control image of size (height, width) with Canny edges of the logo
    placed in the bottom-right corner.  The logo is resized to `region_scale` of
    the smaller image dimension, and offset in by `margin_scale`.
    """
    # determine region size and margins based on image dims
    region_size = int(min(width, height) * region_scale)
    margin     = int(min(width, height) * margin_scale)

    # 1) resize logo, grayscale & Canny
    logo_small = logo.resize((region_size, region_size))
    logo_np    = np.array(logo_small)
    logo_gray  = cv2.cvtColor(logo_np, cv2.COLOR_RGB2GRAY)
    logo_canny = cv2.Canny(logo_gray, 100, 200)

    # 2) blank canvas & paste canny patch at bottom-right
    canvas     = np.zeros((height, width), dtype=np.uint8)
    x_off      = width  - region_size - margin
    y_off      = height - region_size - margin
    canvas[y_off:y_off+region_size, x_off:x_off+region_size] = logo_canny

    return Image.fromarray(canvas).convert("RGB")


def place_patch_bottom_right(base, patch, margin=30):
    h, w = patch.shape[:2]
    H, W = base.shape[:2]
    x_off = W - w - margin
    y_off = H - h - margin
    base[y_off:y_off + h, x_off:x_off + w] = patch
    return base

def prepare_control_and_mask(logo_path, height, width, size=(224, 224), margin=30):
    logo = Image.open(logo_path).convert("RGBA").resize(size)
    logo_np = np.array(logo)

    # Create Canny from RGB
    gray = cv2.cvtColor(logo_np[:, :, :3], cv2.COLOR_RGB2GRAY)
    canny = cv2.Canny(gray, 100, 200)
    control_canvas = np.zeros((height, width), dtype=np.uint8)
    control_canvas = place_patch_bottom_right(control_canvas, canny, margin)

    # Create mask from alpha
    alpha = logo_np[:, :, 3]
    mask_canvas = np.zeros((height, width), dtype=np.uint8)
    mask_canvas = place_patch_bottom_right(mask_canvas, alpha, margin)

    control_image = Image.fromarray(control_canvas).convert("RGB")
    mask_image = Image.fromarray(mask_canvas)
    return control_image, mask_image



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

def dynamic_adapter_callback_creator(adapter_scale_default, steps):     # not in use 
    """
    Returns a callback function that updates the IP-adapter scale dynamically.
    The scale decays linearly from the provided starting value to 0 over the number of steps.
    """
    def dynamic_adapter_callback(pipe_obj, step: int, timestep: float, callback_kwargs, **kwargs):
        # Decay: from adapter_scale_default down to 0 linearly
        new_scale = adapter_scale_default * max(0.0, 1 - step / float(steps))
        pipe_obj.set_ip_adapter_scale(new_scale)
        #print(f"Step {step}/{steps}: Adapter scale = {new_scale:.3f}")
        # Return an empty dict to satisfy pipeline expectations.
        return {}
    return dynamic_adapter_callback


# Hard cutoff: Adapter at full strength up to cutoff, then zero.
def hard_cutoff_adapter_callback_creator(adapter_scale_default, steps, cutoff_ratio):
    """
    Returns a callback that applies the IP-adapter with full strength up to a hard cutoff point.
    After the cutoff step, the adapter influence is immediately set to 0.

    Parameters:
    - adapter_scale_default: Full strength value before cutoff.
    - steps: Total number of inference steps.
    - cutoff_ratio: Fraction of total steps after which adapter should be cut off.
    """
    cutoff_step = int(steps * cutoff_ratio)

    def callback(pipe_obj, step: int, timestep: float, callback_kwargs, **kwargs):
        scale = adapter_scale_default if step < cutoff_step else 0.0
        pipe_obj.set_ip_adapter_scale(scale)
        return {}

    return callback


def dynamic_adapter_callback_creator_cutoff(adapter_scale_default, steps, active_ratio=0.2, min_scale=0.0):
    """
    Returns a callback that applies the IP-adapter strongly at first, then linearly drops off 
    to `min_scale` after a specified active_ratio of the total steps.

    Parameters:
    - adapter_scale_default: Initial IP-adapter scale.
    - steps: Total number of diffusion steps.
    - active_ratio: Fraction of steps during which the adapter is fully applied.
    - min_scale: The adapter scale after cutoff. Use >0 to retain influence, 0.0 to fully disable.
    """
    active_steps = int(steps * active_ratio)

    def dynamic_adapter_callback(pipe_obj, step: int, timestep: float, callback_kwargs, **kwargs):
        if step < active_steps:
            scale = adapter_scale_default
        else:
            # Linearly decay to min_scale over the next active_steps (cutoff duration)
            decay_steps = active_steps
            decay_progress = (step - active_steps) / max(1, decay_steps)
            scale = adapter_scale_default * (1 - decay_progress) + min_scale * decay_progress
        pipe_obj.set_ip_adapter_scale(scale)
        return {}

    return dynamic_adapter_callback


def create_ip_adapter_mask(
    control_image: Image.Image,
    height: int,
    width: int,
    blurred: bool = False,
    blur_scale: float = 0.1
) -> Image.Image:
    """
    From the control_image’s nonzero region (i.e. the Canny patch), build either:
      • a hard rectangle mask, or
      • a soft circular mask with extra padding = blur_scale * region_size.
    Ensures the mask exactly lines up with the Canny block generated above.
    """
    
    control_np = np.array(control_image.convert("L"))
    coords     = cv2.findNonZero(control_np)

    if coords is None:
        raise RuntimeError("No edges found in control image—check Canny thresholds.")


    x, y, w, h = cv2.boundingRect(coords)
    x0, y0     = x, y
    x1, y1     = x + w, y + h

    if not blurred:
        # hard rectangle
        mask = np.zeros((height, width), dtype=np.uint8)
        cv2.rectangle(mask, (x0, y0), (x1, y1), 255, thickness=-1)

    else:
        # soft circular mask centered on that block
        cx = (x0 + x1) // 2
        cy = (y0 + y1) // 2
        pad = int(max(w, h) * blur_scale)
        radius = max(w, h) // 2 + pad

        Y, X = np.ogrid[:height, :width]
        dist = np.sqrt((X - cx)**2 + (Y - cy)**2)
        soft = np.clip(1 - (dist / radius), 0, 1)
        mask = (soft * 255).astype(np.uint8)

    return Image.fromarray(mask)


# New function: create_ip_adapter_mask_from_alpha
def create_ip_adapter_mask_from_alpha(logo_path, height, width, resize_to=(224, 224)):
    logo = Image.open(logo_path).convert("RGBA").resize(resize_to)
    alpha = np.array(logo.split()[-1])  # Alpha channel
    mask = np.zeros((height, width), dtype=np.uint8)

    # Position in bottom-right
    h, w = alpha.shape
    x_off = width - w - 30  # match margin
    y_off = height - h - 30
    mask[y_off:y_off+h, x_off:x_off+w] = alpha
    return Image.fromarray(mask)


def run_grid_search(
    pipe, prompt, negative_prompt, steps,
    grid_guidance, grid_adapter, grid_cn, grid_cutoffs,
    image_embeds, control_image, ip_adapter_masks,
    seed, device, grid_dir, height, width
):
    results_info = []
    for g, a, cn, co in itertools.product(grid_guidance, grid_adapter, grid_cn, grid_cutoffs):
        print(f"→ G={g}, A={a}, CN={cn}, cutoff={co:.2f}")
        pipe.set_ip_adapter_scale(a)
        dynamic_cb = hard_cutoff_adapter_callback_creator(a, steps, cutoff_ratio=co)
        gen = torch.Generator(device=device).manual_seed(seed)

        out = pipe(
            prompt=prompt,
            ip_adapter_image_embeds=image_embeds,
            ip_adapter_masks=ip_adapter_masks,
            image=control_image,
            negative_prompt=negative_prompt,
            num_inference_steps=steps,
            guidance_scale=g,
            controlnet_conditioning_scale=cn,
            height=height,
            width=width,
            generator=gen,
            callback_on_step_end=dynamic_cb,
            callback_steps=1
        )
        img = out.images[0]

        fname = f"res_g{g}_A{a}_CN{cn}_CO{int(co*100)}_SS{steps}.png"
        fpath = os.path.join(grid_dir, fname)
        img.save(fpath)
        print(f"  saved {fname}")

        # store: (image, guidance, adapter, cn, cutoff, path)
        results_info.append((img, g, a, cn, co, fpath))

        del img, gen
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return results_info


def generate_collages(
    results_info,
    grid_guidance, grid_adapter, grid_cn, grid_cutoffs,
    width, height, grid_dir
):
    combo_list = sorted((a, cn) for a in grid_adapter for cn in grid_cn)
    num_rows, num_cols = len(grid_guidance), len(combo_list)

    for co in grid_cutoffs:
        # filter just this cutoff
        subset = [r for r in results_info if r[4] == co]

        fig, axs = plt.subplots(num_rows, num_cols, figsize=(num_cols*3, num_rows*3))
        fig.subplots_adjust(hspace=0.4, wspace=0.2)
        if num_rows == 1: axs = [axs]
        if num_cols == 1: axs = [[ax] for ax in axs]

        col_map = {combo: idx for idx, combo in enumerate(combo_list)}
        for img, g, a, cn, _, _ in subset:
            row = grid_guidance.index(g)
            col = col_map[(a, cn)]
            ax  = axs[row][col]
            ax.imshow(np.array(img)); ax.axis("off")
            ax.set_title(f"G:{g}, A:{a}, CN:{cn}", fontsize=9, pad=4, backgroundcolor="white")

        outname = os.path.join(grid_dir, f"collage_cutoff{int(co*100)}.png")
        plt.savefig(outname, bbox_inches="tight", dpi=300)
        plt.close(fig)
        print(f"▶ Saved collage for cutoff={co:.2f} → {outname}")


def cleanup(pipe, controlnet, image_embeds, device):
    """Clears the main objects from memory and empties the appropriate GPU/accelerator cache."""
    del pipe, controlnet, image_embeds
    gc.collect()

    if device == "cuda":
        torch.cuda.empty_cache()
    elif device == "mps" and hasattr(torch, "mps") and torch.backends.mps.is_available():
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
    steps = 50
    height = 1024 #lowering img resolution from 384 to 256
    width = 2048 #lowering from 768 to 512

    # Grid search parameters:
    # **UPDATE:** grid_adapter is now a list of starting adapter scales.
    # initial run
    # grid_adapter = [0.1, 0.3, 0.6, 1.0]
    # grid_guidance = [6.0]
    # grid_cn = [1.0, 0.7]
    # grid_cutoffs  = [1/10, 1/5, 1/3]

    # second run: 
    grid_adapter = [0.05, 0.1, 0.15, 0.2, 0.3]
    grid_cn = [0.7]
    grid_cutoffs = [1/20, 1/10, 0.15, 1/5, 1/3]
    grid_guidance = [6.0]

    # For pipeline setup, use the first adapter value.
    adapter_scale_default = grid_adapter[0]

    try_mps = True # for mac setup, not in use for CUDA
    device = get_device()
    dtype = get_dtype(device)

    # === Load Logo and Prepare Images ===

    # Prepare control (Canny) image at bottom-right, dynamically scaled:
    logo_path = "logos/ipsum.png"
    logo_lowres = load_logo(logo_path, size=(224, 224))
    control_image, logo_mask = prepare_control_and_mask(logo_path, height, width)

    # === IP-Adapter Masking ===
    # Create IP-Adapter mask that matches that exact region, with a soft edge:
    # logo_mask = create_ip_adapter_mask(
    #     control_image,
    #     height,
    #     width,
    #     blurred=True,
    #     blur_scale=0.1         # reduced from 0.2 to better match control image
    # )

    # Use alpha mask approach for logo masking:
    # logo_mask = create_ip_adapter_mask_from_alpha(logo_path, height, width, resize_to=(224, 224


    processor = IPAdapterMaskProcessor()
    ip_adapter_masks = processor.preprocess([logo_mask], height=height, width=width)


        # === Setup Pipeline ===
    pipe, controlnet = setup_pipeline(
        model_id,
        ip_adapter_path,
        ip_weights,
        subfolder,
        controlnet_model_id,
        device,
        dtype,
        adapter_scale_default
    )

    # Only enable xformers if running on CUDA
    if device == "cuda":
        pipe.enable_xformers_memory_efficient_attention()

    # Prepare image embeddings
    logo_highres = load_logo(logo_path, size=(768, 768))  # was (512, 512)
    image_embeds = prepare_image_embeds(pipe, logo_highres, device)

    # === Grid Search Setup ===
    grid_dir = get_grid_directory("results/grid_search")
    # Note: dynamic_callback will be created inside run_grid_search for each adapter value.

    control_image.save("debug_canny_input_forrest.png")
    logo_mask.save("debug_mask_forrest.png") 
    # Run grid search and collect results.
    results_info = run_grid_search(
        pipe, prompt, negative_prompt, steps,
        grid_guidance, grid_adapter, grid_cn, grid_cutoffs,
        image_embeds, control_image, ip_adapter_masks,
        seed, device, grid_dir, height, width
    )

    #control_image.save("debug_canny_input.png")
    #logo_mask.save("debug_mask.png") 

    # Generate collages from the grid search results.
    generate_collages(
        results_info,
        grid_guidance, grid_adapter, grid_cn, grid_cutoffs,
        width, height, grid_dir
    )

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
    cleanup(pipe, controlnet, image_embeds, device)

if __name__ == "__main__":
    main()

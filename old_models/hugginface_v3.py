import torch
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel
from controlnet_aux import CannyDetector
from PIL import Image
import matplotlib.pyplot as plt
import os


# CONFIGS

# Prompt setup
prompt = "A banner image of a summer forest, vibrant colors"


dtype = torch.float32   # Global dtype for numerical stability on MPS
steps = 5              # Number of inference steps
seeds = [42, 123, 999]  # Seeds

# Models (compatible and publicly accessible)
models = {
    "Stable Diffusion v1.5": "runwayml/stable-diffusion-v1-5",
    "Openjourney": "prompthero/openjourney"
}

print(torch.__version__)
print(torch.backends.mps.is_available())
print(torch.backends.mps.is_built())

# Device setup explicitly for Apple Silicon (MPS)
device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using device: {device}")

# ControlNet setup (Canny)
controlnet = ControlNetModel.from_pretrained(
    "lllyasviel/sd-controlnet-canny",
    torch_dtype=dtype,
    use_safetensors=True
).to(device)

# Pipeline kwargs
pipe_kwargs = {
    "torch_dtype": dtype,
    "use_safetensors": True,
    "safety_checker": None          # AVOIDS image generations beeing filtered to NSFW and sensored
}

# Canny edge detector, current segmentation mask attempt
canny = CannyDetector()

# Input/control image preprocessing
control_image = Image.open("nike.png").resize((512, 512))
control_image_canny = canny(control_image)
control_image_canny.save("canny_test.png")


def make_generator(seed):
    return torch.Generator(device=device).manual_seed(seed)

# Image generation
results = {}
for model_name, model_path in models.items():
    print(f"\nGenerating images with model: {model_name}")
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        model_path,
        controlnet=controlnet,
        **pipe_kwargs
    ).to(device)

    pipe.enable_attention_slicing()

    model_images = []
    for seed in seeds:
        generator = make_generator(seed)

        image = pipe(
            prompt,
            num_inference_steps=steps,
            guidance_scale=7.5,
            image=control_image_canny,
            generator=generator
        ).images[0]

        # Save
        filename = f"{model_name.replace(' ', '_')}_seed_{seed}.png"
        image.save(filename)
        print(f"✅ Saved: {filename}")

        model_images.append(image)

    results[model_name] = model_images

#  Plot results
fig, axes = plt.subplots(len(models), len(seeds), figsize=(5 * len(seeds), 5 * len(models)))

for i, (model_name, images) in enumerate(results.items()):
    for j, img in enumerate(images):
        ax = axes[i, j] if len(models) > 1 else axes[j]
        ax.imshow(img)
        ax.axis('off')
        ax.set_title(f"{model_name}\nSeed: {seeds[j]}")


os.makedirs("results/model_comparisons", exist_ok=True)

plt.tight_layout()
plt.savefig("results/model_comparisons/model_comparison_v3.png", dpi=300)
plt.show()
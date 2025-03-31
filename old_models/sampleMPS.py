from diffusers import StableDiffusionPipeline
import torch

device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using device: {device}")

model_id = "runwayml/stable-diffusion-v1-5"
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
pipe = pipe.to(device)

prompt = "one small blue ball to the left, and an big orange to the right"
image = pipe(prompt).images[0]

image.save("test_output.png")
print("✅ Image generation successful!")
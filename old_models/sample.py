from diffusers import StableDiffusionPipeline
import torch

model_id = "runwayml/stable-diffusion-v1-5"
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
pipe.to("cuda" if torch.cuda.is_available() else "cpu")

prompt = "A quick test of stable diffusion"
image = pipe(prompt).images[0]

image.save("test_output.png")
print("✅ Image generation successful!")


# from diffusers import DiffusionPipeline

# pipeline = DiffusionPipeline.from_pretrained("stable-diffusion-v1-5/stable-diffusion-v1-5", use_safetensors=True)


# pipeline.to("cuda")

# image = pipeline("An image of a squirrel in Picasso style").images[0]
# image
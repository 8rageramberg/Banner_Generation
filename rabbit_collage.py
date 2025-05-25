import os
import matplotlib.pyplot as plt
from PIL import Image

# Define your image folders
folders = [
    #"final_results/grid_search_19_forrest_fox",
    # "final_results/grid_search_20_beach_fox",
    # "final_results/grid_search_23_purple_star",
    # "final_results/grid_search_24_white_star",
    "final_results/grid_search_25_beach_logoipsum"
]

# Helper to get top images with A and CO emphasis
def get_representative_images(folder, allowed_adapters, allowed_cutoffs):
    matches = []
    for file in sorted(os.listdir(folder)):
        if not file.endswith(".png"):
            continue

        parts = file.split("_")
        adapter = cutoff = None
        for part in parts:
            if part.startswith("A"):
                adapter = "A" + part[1:].split("-")[0]
            if part.startswith("CO"):
                cutoff = "CO" + part[2:].split("-")[0]

        if adapter in allowed_adapters and cutoff in allowed_cutoffs:
            matches.append(os.path.join(folder, file))

    def extract_params(filename):
        adapter = cutoff = 0
        parts = filename.split("_")
        for part in parts:
            if part.startswith("A"):
                try:
                    adapter = float(part[1:])
                except:
                    pass
            if part.startswith("CO"):
                try:
                    cutoff = int(part[2:])
                except:
                    pass
        return (adapter, cutoff)

    matches.sort(key=lambda path: extract_params(os.path.basename(path)))
    return matches

allowed_adapters = {"A0.05", "A0.1", "A0.15", "A0.2", "A0.3"}
allowed_cutoffs = {"CO5", "CO10", "CO15", "CO20", "CO33"}

# Gather images
images = []
for folder in folders:
    images.extend(get_representative_images(folder, allowed_adapters, allowed_cutoffs))

import math

num_images = len(images)
ncols = 5
nrows = 5

fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(20, 15))

# Add global information as a second title line
prompt_info = (
    "Prompt: A clean minimalistic product banner with soft lighting and white background,\n"
    "centered skincare product, natural shadows, stylized rabbit logo in the bottom-right corner."
)
global_settings = "Global params: G=6.0, CN=0.7, Steps=50"

fig.text(0.5, 0.96, prompt_info, ha='center', fontsize=12)
fig.text(0.5, 0.935, global_settings, ha='center', fontsize=11)

for ax, img_path in zip(axes.flatten(), images):
    img = Image.open(img_path).resize((1200, 600))
    ax.imshow(img)
    filename = os.path.basename(img_path).replace("res_", "").replace(".png", "")
    # Remove redundant global parameters from the title
    for param in ["g6.0", "CN0.7", "SS50"]:
        filename = filename.replace(param, "")
    ax.set_title(filename.replace("_", " ").strip(", "), fontsize=10)
    ax.axis("off")

for ax in axes.flatten()[len(images):]:
    ax.axis("off")

plt.subplots_adjust(left=0.01, right=0.99, top=0.92, bottom=0.01, wspace=0.01, hspace=0.01)
plt.savefig("final_results_collages/rabbit_grid_collage_5x5.png", dpi=300, bbox_inches="tight")
plt.show()
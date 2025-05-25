# Stable Diffusion XL Grid Generator with ControlNet and IP-Adapter

This project performs automated image generation using [Stable Diffusion XL](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0) with **ControlNet** and **IP-Adapter**. It generates high-resolution images with logos embedded in the bottom-right corner using Canny edge guidance and embedding-based control.


- **Grid Search Tuning**: Automatically explores combinations of key generation parameters:
  - `grid_adapter`: Strength of the IP-Adapter (e.g., `[0.1, 1.0]`) — controls the semantic influence of the embedded logo.
  - `grid_cn`: ControlNet conditioning strength (e.g., `[0.1, 1.0]`) — governs how strictly structural guidance (e.g., logo placement) is followed.
  - `grid_guidance`: Prompt guidance scale (e.g., `[5.0, 9.0]`) — determines how strongly the text prompt influences generation.
  - `grid_cutoffs`: Step cutoffs (e.g., `[1/10, 1/3]`) — defines at which point during diffusion the IP-Adapter influence is reduced or turned off.
  - `steps`: Number of inference steps (e.g., `50+`) — higher values typically yield more detailed results.
  - `width`, `height`: Image resolution (e.g., `2048×1024`) — controls the dimensions of generated images.



## Requirements

- Python 3.10+
- CUDA-compatible GPU with **≥16 GB VRAM** recommended (e.g., A100, V100, RTX 3090)
- Host machine with **≥32 GB RAM** for smooth operation

## Weights
This project uses pre-trained weights from:
	•	Stable Diffusion XL: stabilityai/stable-diffusion-xl-base-1.0
	•	ControlNet: diffusers/controlnet-canny-sdxl-1.0
	•	IP-Adapter weights: stored locally in h94/IP-Adapter/ip-adapter_sdxl.bin


## Environment
To recreate the exact development environment:
	1.	Make sure Conda is installed.
	2.	Run the following commands from your terminal:

conda env create -f full_conda_env.yml
conda activate torchenv

The full_conda_env.yml file was generated from the active environment using:
conda env export > full_conda_env.yml

Why Conda?
Conda was chosen after significant troubleshooting with cluster compatibility and dependencies:
	•	It handles binary dependencies (e.g., CUDA, cuDNN) more gracefully than pip alone.
	•	It’s better suited for managing complex ML toolchains like PyTorch, torchvision, and diffusers with GPU support.
	•	It works well on shared compute clusters where system-level packages can’t be modified.
	•	It allows us to lock down exact versions across platforms using an env.yml file.
	•	Some partitions had hardware compatibility issues with specific versions of PyTorch/CUDA — Conda let us pin versions and avoid runtime crashes.


## Testing

A test script (`check_cuda.py`) is provided to verify some of the environment pipeline setup. You can run it to confirm environment and CUDA availability. No model training is needed — the pipeline uses pre-trained weights for all generation tasks.

If your project requires `test.py` (e.g., to validate outputs or reproduce a reported result), include that script and explain it here.


## Project Overview


Directory structure:

- `final_model.py` – Main generation script with grid search
- `bot.py` – Telegram bot integration for remote generation
- `full_conda_env.yml` – Conda environment export
- `check_cuda.py` – Simple CUDA availability test
- `dqx2q_final.sbatch` – Example SLURM batch script (requires `logs/`)
- `CLIP_score_test/` – Unused CLIP score test code
- `logos/` – Input logos for IP-Adapter
- `results/` – Output folder for generations
- `README.md` – This file
- `LICENSE` – Project license

## Safety Considerations

This project uses Stable Diffusion XL with ControlNet and IP-Adapter. By default, the Hugging Face `diffusers` pipeline includes an NSFW safety checker designed to flag or suppress potentially inappropriate content.

In our case, the safety checker was sometimes found to interfere with valid, non-offensive outputs (e.g., certain colors, shapes, or logo features).

If needed for your use case, the safety checker can be **disabled** by uncommenting a single line in `final_model.py`. See the `setup_pipeline()` function for guidance.

> Disabling the safety checker may allow undesired or unsafe content to be generated. Use responsibly.

Read more here: [CompVis/stable-diffusion-safety-checker](https://huggingface.co/CompVis/stable-diffusion-safety-checker)

## License
This project is licensed under the Apache License 2.0. See the LICENSE file for details.
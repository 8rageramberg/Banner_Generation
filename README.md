# Banner_Generation


# installation
https://github.com/huggingface/diffusers



conda install pytorch torchvision torchaudio -c pytorch -y
pip install diffusers transformers accelerate safetensors controlnet_aux Pillow

conda install pytorch torchvision torchaudio -c pytorch-nightly -y (metal performance shader for apple silicone)



Step-by-step Pipeline:

1. Input Logo & Prompt
	•	User uploads the logo image and provides a text prompt.
	•	Example: "Summer-themed forest banner for DNB, logo in top-right corner."

2. Preprocess Logo Image
	•	Resize and position the logo exactly where desired on a blank canvas or transparent mask (e.g., top-right corner).

3. Conditioning via ControlNet
	•	Use the prepared logo image for ControlNet conditioning.
	•	ControlNet instructs Stable Diffusion to preserve the position and integrity of the logo while generating the surrounding image according to the provided prompt.

4. Generate Image
	•	Run the diffusion pipeline (Diffusers + ControlNet) to produce the final image.

5. Postprocessing (Optional)
	•	Perform an inpainting step if needed to seamlessly blend the logo with the generated content.



Recommended Datasets for Fine-tuning and Evaluation:

You likely won’t need large-scale training from scratch unless you aim for a research-grade solution. Typically, fine-tuning on smaller datasets is sufficient. However, if you’re looking for datasets for evaluation or fine-tuning, consider:

1. COCO (Common Objects in Context)
	•	Provides rich annotations and high-quality images.
	•	Excellent for evaluating semantic correctness and object placements.
	•	Link to COCO dataset

2. LAION-5B Subset
	•	Stable Diffusion was pre-trained on LAION datasets.
	•	Use subsets like LAION-5B or LAION-Aesthetics to evaluate model generalization.

3. Logo/Brand Datasets
	•	LogoDet-3K: For training and testing logo placement and consistency.
	•	FlickrLogos-47: Annotated logo data suitable for testing detection, recognition, and accurate placement.





# MPS configs

 Recommended “best-practice” MPS Configuration:

According to Hugging Face (as you’ve pasted), the ideal setup for Stable Diffusion on Apple Silicon (MPS backend) is:
	•	macOS 13+ (recommended)
	•	PyTorch 2.0 or later (recommended)
	•	Using MPS backend explicitly via .to("mps")
	•	Enable Attention Slicing for memory efficiency and performance.




# Notes to Hugginface models

hugginface_v1: 

quick generation, displaying good results with the current nike logo showing, with elements of the original prompt


hugginface_v2: 

decent generation, tried implementing MPS on float16, but resulted in complete black generated images.

- stablediffusion v 1.5 however did decent, have currently not seen results from the model tho: 

Generating images with model: Stable Diffusion v1.5
100%|██████████| 1/1 [00:09<00:00,  9.65s/it]
100%|██████████| 20/20 [00:42<00:00,  2.12s/it]
100%|██████████| 20/20 [00:37<00:00,  1.86s/it]
100%|██████████| 20/20 [00:35<00:00,  1.76s/it]

Generating images with model: Openjourney
- Tried up the dtype to float32, resulting in insaly slow generation for the Openjourney model, unsure if i bother run the rest....
100%|██████████| 1/1 [01:24<00:00, 84.21s/it]
100%|██████████| 20/20 [26:02<00:00, 78.13s/it]
 30%|███       | 6/20 [06:39<14:34, 62.46s/it] 
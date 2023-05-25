# Joint Mask DDPM

This is a joint mask-image generation model, which is able to accomplish 3 main tasks:

1. Labelled dataset generation (joint generation of mask and image)
2. Image segmentation (mask generation conditioned by image)
3. Mask-conditioned image generation

Ongoing...

We only provide API segmenting Prostate MRI dataset (```main_segment.py```). For custom dataset, please create a pytorch dataset object in config file, and replace the Prostate MRI dataset with the custom dataset.
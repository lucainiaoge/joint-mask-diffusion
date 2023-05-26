# Image-mask Diffusion (IMDiff)

This is a joint mask-image generation model, which is able to accomplish 3 main tasks:

1. Labelled dataset generation (joint generation of mask and image)
2. Image segmentation (mask generation conditioned by image)
3. Mask-conditioned image generation

# Data Preparation

Download ProstateMRI dataset from [https://liuquande.github.io/SAML/](https://liuquande.github.io/SAML/), and unzip at the root folder, creating ```./ProstateMRI```.

# Training

```
python main_train.py 
--ckpt-dir [checkpoint_save_dir] 
--milestone [int, loading checkpoint number] 
--loss-conf [choose from "dice" and "mse"] 
--batch-size [int, default: 8] 
--train-steps [int, default: 500000] 
--save-interval [int, default: 10000]
--lr [float, default: 0.0001]
```


# Testing

To generate image-label dataset, run

```
python main_test.py
--milestone [int, loading checkpoint number]  
--load-dir [checkpoint dir] 
--save-dir [generated dataset dir] 
--num-samples [int] 
```

Similarly, run ```python main_segment.py``` to implement semantic segmentation on Prostate MRI dataset.

We only provide API (```main_segment.py```) segmenting Prostate MRI dataset. For custom dataset, please create a pytorch dataset object in config file, and replace the Prostate MRI dataset with the custom dataset.
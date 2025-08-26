# Mask-Based Multi-Style Transfer  (Ours)

<img src="./output_dk/ours_multi.png" width="300">

# SetUP

## Create a Conda Environment

```
conda env create -f environment.yaml
conda activate StyleID
```

## Download StableDiffusion Weights

Download <a href='https://huggingface.co/CompVis/stable-diffusion-v-1-4-original/tree/main'>Model weight (sd-v1-4.ckpt)</a> and put it in the `models/ldm/stable-diffusion-v1/model.ckpt` **(rename: `sd-v1-4.ckpt` > `model.ckpt`)**

## Description of Folders

- `data_styleid`: dataset of existing StyleID
- `data_vis`: **datset for Our inference**
    - `/cnt`: content images and `mask.npy`
    - `/sty`: style images
- `dataset_ours`: our dataset
- `output_dk`: output dir
- `precomputed_feats_k`: `feature map.pkl` after DDIM Inversion

# Inference

## DDIM Inversion for saving feature map

is saved in `precomputed_feats_k` dir

```
python generate_pkl_only.py --cnt data_vis/cnt --sty data_vis/sty
```

## Content Mask Generation

### Option 1. FastSAM

```
Developing ...
```

### Option 2. Gradio

```
python gradio_make_mask.py
```

## Create `mask.npy` for Content Image only

```
python make_mask_npy.py --mask_pth ./dataset_ours/mask/content_flower_mask.png
```

## Copy 
Put the content image you want to inference in `dataset_ours/contents` and `.npy` corresponding in `dataset_ours/mask` into `data_vis/cnt` and the style image into `data_vis/style`

```
data_vis
├── cnt
│   ├── {content}.jpg
│   └── {content_mask}.npy
└── sty
    ├── {style-1}.jpg
    └── {style-2}.jpg
```


## Inference

- Gamma: Query preservation
- T: Tau (variance of attention map)
- Ratio: High-frequency radius ratio

```
python run_ori.py --cnt data_vis/cnt --sty data_vis/sty --gamma 0.5 --T 1.5 --ratio 0.5
```

Check `output_dk` dir!!!
import argparse
import os
import torch
from torch import autocast
import numpy as np
from omegaconf import OmegaConf
from PIL import Image
from einops import rearrange
from pytorch_lightning import seed_everything
from contextlib import nullcontext
import copy
import torchvision.transforms as transforms
import pickle
import time

from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler

# # residual injection
#from high_frequency_final import high_pass_filter

feat_maps = []

def load_img(path):
    image = Image.open(path).convert("RGB")
    x, y = image.size
    print(f"Loaded input image of size ({x}, {y}) from {path}")
    h = w = 512
    image = transforms.CenterCrop(min(x, y))(image)
    image = image.resize((w, h), resample=Image.Resampling.LANCZOS)
    image = np.array(image).astype(np.float32) / 255.0
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image)
    return 2. * image - 1.

def load_model_from_config(config, ckpt, verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)

    model.cuda()
    model.eval()
    return model

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cnt', default = None, help='Content image folder path')
    parser.add_argument('--sty', default = None, help='Style image folder path')
    parser.add_argument('--ddim_inv_steps', type=int, default=50)
    parser.add_argument('--save_feat_steps', type=int, default=50, help='DDIM eta')
    parser.add_argument('--start_step', type=int, default=49)
    parser.add_argument('--ddim_eta', type=float, default=0.0)
    parser.add_argument('--H', type=int, default=512)
    parser.add_argument('--W', type=int, default=512)
    parser.add_argument('--C', type=int, default=4)
    parser.add_argument('--f', type=int, default=8)
    parser.add_argument("--attn_layer", type=str, default='6,7,8,9,10,11', help='injection attention feature layers')
    parser.add_argument('--model_config', type=str, default='models/ldm/stable-diffusion-v1/v1-inference.yaml')
    parser.add_argument('--precomputed', type=str, default='./precomputed_feats_k')
    #parser.add_argument('--precomputed', type=str, default='./precomputed_feats_k')
    parser.add_argument('--ckpt', type=str, default='models/ldm/stable-diffusion-v1/model.ckpt')
    parser.add_argument('--precision', type=str, default='autocast', help='choices: ["full", "autocast"]')
    opt = parser.parse_args()

    seed_everything(22)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    feat_path_root = opt.precomputed
    os.makedirs(feat_path_root, exist_ok=True)

    model_config = OmegaConf.load(opt.model_config)
    model = load_model_from_config(model_config, opt.ckpt)
    model = model.to(device)
    unet_model = model.model.diffusion_model
    sampler = DDIMSampler(model)
    
    self_attn_output_block_indices = list(map(int, opt.attn_layer.split(',')))
    ddim_inversion_steps = opt.ddim_inv_steps
    save_feature_timesteps = ddim_steps = opt.save_feat_steps
    

    sampler.make_schedule(ddim_num_steps=ddim_steps, ddim_eta=opt.ddim_eta, verbose=False) 
    time_range = np.flip(sampler.ddim_timesteps)
    idx_time_dict = {}
    time_idx_dict = {}
    for i, t in enumerate(time_range):
        idx_time_dict[t] = i
        time_idx_dict[i] = t

    seed = torch.initial_seed()
    opt.seed = seed
    
    global feat_maps
    feat_maps = [{'config': {
                'T': 1.5
                }} for _ in range(50)]

    def ddim_sampler_callback(pred_x0, xt, i):
        save_feature_maps_callback(i)
        save_feature_map_z(xt, 'z_enc', i)
    
    def save_feature_map(feature_map, filename, time):
        global feat_maps
        cur_idx = idx_time_dict[time]
        feat_maps[cur_idx][f"{filename}"] = feature_map

    def save_feature_maps(blocks, i, feature_type="input_block"):
        block_idx = 0
        for block_idx, block in enumerate(blocks):
            if len(block) > 1 and "SpatialTransformer" in str(type(block[1])):
                if block_idx in self_attn_output_block_indices:
                    # self-attn
                    q = block[1].transformer_blocks[0].attn1.q
                    k = block[1].transformer_blocks[0].attn1.k
                    v = block[1].transformer_blocks[0].attn1.v
                    save_feature_map(q, f"{feature_type}_{block_idx}_self_attn_q", i)
                    save_feature_map(k, f"{feature_type}_{block_idx}_self_attn_k", i)
                    save_feature_map(v, f"{feature_type}_{block_idx}_self_attn_v", i)
            block_idx += 1
            
    def save_feature_maps_callback(i):
        save_feature_maps(unet_model.output_blocks , i, "output_block")

    def save_feature_map_z(xt, name, time):
        global feat_maps
        cur_idx = idx_time_dict[time]
        feat_maps[cur_idx][name] = xt

    def residual_injection_callback(pred_x0, xt, t):
        # feature map 저장
        save_feature_maps_callback(t)
        save_feature_map_z(xt, 'z_enc', t)

        t_int = int(t)
        if t_int not in residuals_all:
            residuals_all[t_int] = {}

        for block_id in range(6, 12):
            if block_id >= len(unet_model.output_blocks):
                break

            for module in reversed(unet_model.output_blocks[block_id]):
                if module.__class__.__name__.endswith("ResBlock"):
                    if hasattr(module, 'out_skip') and module.out_skip is not None:
                        skip = module.out_skip.detach().cpu()
                        #skip_hf = high_pass_filter(skip, radius=6)
                        key_skip = f"output_block_{block_id}_cnt_skip"
                        residuals_all[t_int][key_skip] = skip
                        print(f"[Callback] t={t_int}, saved {key_skip}")

                    if hasattr(module, 'out_h') and module.out_h is not None:
                        h = module.out_h.detach().cpu()
                        #h_hf = high_pass_filter(h, radius=6)
                        key_h = f"output_block_{block_id}_cnt_h"
                        residuals_all[t_int][key_h] = h
                        print(f"[Callback] t={t_int}, saved {key_h}")

                    if hasattr(module, 'out_merged') and module.out_merged is not None:
                        key_full = f"output_block_{block_id}_residual"
                        residuals_all[t_int][key_full] = module.out_merged.detach().cpu()
                        print(f"[Callback] t={t_int}, saved {key_full}")

                    break  # 마지막 ResBlock만 처리

                    
    start_step = opt.start_step
    precision_scope = autocast if opt.precision=="autocast" else nullcontext
    uc = model.get_learned_conditioning([""])
    shape = [opt.C, opt.H // opt.f, opt.W // opt.f]
    sty_img_list = sorted(os.listdir(opt.sty))
    cnt_img_list = sorted(os.listdir(opt.cnt))

    # === STYLE IMAGES ===
    if opt.sty is not None and os.path.exists(opt.sty):
        sty_img_list = sorted(os.listdir(opt.sty))
        for sty_name in sty_img_list:
            sty_path = os.path.join(opt.sty, sty_name)
            init_sty = load_img(sty_path).to(device)
            sty_feat_name = os.path.join(feat_path_root, os.path.splitext(sty_name)[0] + '_sty.pkl')

            if os.path.isfile(sty_feat_name):
                print(f"Precomputed style feature exists: {sty_feat_name}")
            else:
                init_sty_latent = model.get_first_stage_encoding(model.encode_first_stage(init_sty))
                sty_z_enc, _ = sampler.encode_ddim(
                    init_sty_latent.clone(),
                    num_steps=ddim_inversion_steps,
                    unconditional_conditioning=uc,
                    end_step=time_idx_dict[ddim_inversion_steps - 1 - start_step],
                    callback_ddim_timesteps=save_feature_timesteps,
                    img_callback=ddim_sampler_callback
                )
                sty_feat = copy.deepcopy(feat_maps)
                with open(sty_feat_name, 'wb') as f:
                    pickle.dump(sty_feat, f)
                print(f"Saved style feature: {sty_feat_name}")
    else:
        print("No style images provided or path does not exist. Skipping style inversion.")

    # === CONTENT IMAGES ===
    if opt.cnt is not None and os.path.exists(opt.cnt):
        cnt_img_list = sorted(os.listdir(opt.cnt))
        for cnt_name in cnt_img_list:
            cnt_path = os.path.join(opt.cnt, cnt_name)
            init_cnt = load_img(cnt_path).to(device)
            cnt_feat_name = os.path.join(feat_path_root, os.path.splitext(cnt_name)[0] + '_cnt.pkl')

            if os.path.isfile(cnt_feat_name):
                print(f"Precomputed content feature exists: {cnt_feat_name}")
            else:
                init_cnt_latent = model.get_first_stage_encoding(model.encode_first_stage(init_cnt))
                #save_path = f"/home/dldpfud/hdd/diffusion_model/styleID/precomputed_feats_share/0801/resnet_611_hf/{cnt_name}_residuals.pkl"
                # unet_model.save_residuals = True
                # unet_model.residual_save_path = save_path
                
                residuals_all = {}  # { timestep: { "output_block_{i}_residual": Tensor, ... }, ... }
                cnt_z_enc, _ = sampler.encode_ddim(
                    init_cnt_latent.clone(),
                    num_steps=ddim_inversion_steps,
                    unconditional_conditioning=uc,
                    end_step=time_idx_dict[ddim_inversion_steps - 1 - start_step],
                    callback_ddim_timesteps=save_feature_timesteps,
                    img_callback=residual_injection_callback
                )
                
                all_path = os.path.join(
                    feat_path_root,
                    f"{os.path.splitext(cnt_name)[0]}_residuals_all.pkl"
                )
                os.makedirs(os.path.dirname(all_path), exist_ok=True)
                with open(all_path, "wb") as f:
                    pickle.dump(residuals_all, f)
                
                base_path = os.path.splitext(all_path)[0]
                #visualize_cnt_high_freq_overlay(residuals_all, base_img=init_cnt, save_path=base_path + "_hf_overlay.png")


                cnt_feat = copy.deepcopy(feat_maps)
                with open(cnt_feat_name, 'wb') as f:
                    pickle.dump(cnt_feat, f)
                print(f"Saved content feature: {cnt_feat_name}")
    else:
        print("No content images provided or path does not exist. Skipping content inversion.")

if __name__ == "__main__":
    main()

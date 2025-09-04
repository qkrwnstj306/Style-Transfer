import argparse, os
import torch
import numpy as np
from omegaconf import OmegaConf
from PIL import Image
from einops import rearrange
from pytorch_lightning import seed_everything
from torch import autocast
from contextlib import nullcontext
import copy
import gc  # 가비지 컬렉션 추가

from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler

import torch.nn.functional as F
import time
import pickle

## 마스크 적용
from ldm.modules.attention import CrossAttention
from high_frequency_final import patch_decoder_resblocks_h_and_cnt_hf, make_content_injection_schedule

from dataset import TripleImageDataset

# 전역 변수 제거 - 함수 내부에서 관리하도록 변경
feat_maps = []

def save_img_from_sample(model, samples_ddim, fname):
    with torch.no_grad():  # gradient 계산 방지
        x_samples_ddim = model.decode_first_stage(samples_ddim)
        x_samples_ddim = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)
        x_samples_ddim = x_samples_ddim.cpu().permute(0, 2, 3, 1).numpy()
        x_image_torch = torch.from_numpy(x_samples_ddim).permute(0, 3, 1, 2)
        x_sample = 255. * rearrange(x_image_torch[0].cpu().numpy(), 'c h w -> h w c')
        img = Image.fromarray(x_sample.astype(np.uint8))
        img.save(fname)

def feat_merge_2sty(opt, cnt_feats, sty_feats_1, sty_feats_2, start_step=0):
    feat_maps = [{
        'config': {  
            'gamma': opt.gamma,
            'T': opt.T,
            'timestep': i,
            'cnt_k': None,
            'sty_q': None,
        }
    } for i in range(50)]

    for i in range(len(feat_maps)):
        cnt_feat = cnt_feats[i]
        sty_feat_1 = sty_feats_1[i]
        sty_feat_2 = sty_feats_2[i]
        ori_keys = cnt_feat.keys()

        for ori_key in ori_keys:
            # .detach()를 추가하여 gradient 연결 끊기
            if ori_key.endswith('q'):
                feat_maps[i][ori_key] = cnt_feat[ori_key].detach() if torch.is_tensor(cnt_feat[ori_key]) else cnt_feat[ori_key]
            if ori_key.endswith('k') or ori_key.endswith('v'):
                feat_maps[i][ori_key] = sty_feat_1[ori_key].detach() if torch.is_tensor(sty_feat_1[ori_key]) else sty_feat_1[ori_key]

            if ori_key.endswith('k') or ori_key.endswith('v'):
                feat_maps[i][ori_key + '_cnt'] = cnt_feat[ori_key].detach() if torch.is_tensor(cnt_feat[ori_key]) else cnt_feat[ori_key]

            if ori_key.endswith('q'):
                feat_maps[i][ori_key + '_sty1'] = sty_feat_1[ori_key].detach() if torch.is_tensor(sty_feat_1[ori_key]) else sty_feat_1[ori_key]

            if ori_key.endswith('q') or ori_key.endswith('k') or ori_key.endswith('v'):
                feat_maps[i][ori_key + '_sty2'] = sty_feat_2[ori_key].detach() if torch.is_tensor(sty_feat_2[ori_key]) else sty_feat_2[ori_key]

    return feat_maps

def adain(cnt_feat, sty_feat):
    with torch.no_grad():  # gradient 계산 방지
        cnt_mean = cnt_feat.mean(dim=[0, 2, 3],keepdim=True)
        cnt_std = cnt_feat.std(dim=[0, 2, 3],keepdim=True)
        sty_mean = sty_feat.mean(dim=[0, 2, 3],keepdim=True)
        sty_std = sty_feat.std(dim=[0, 2, 3],keepdim=True)
        return ((cnt_feat-cnt_mean)/cnt_std)*sty_std + sty_mean

def load_model_from_config(config, ckpt, verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    model.load_state_dict(sd, strict=False)
    model.cuda().eval()
    return model

def clear_attention_cache(model):
    """CrossAttention 모듈의 캐시 클리어"""
    for m in model.modules():
        if isinstance(m, CrossAttention):
            if hasattr(m, 'cnt_name'):
                delattr(m, 'cnt_name')
            # 추가적인 캐시 클리어가 필요한 경우
            if hasattr(m, '_saved_tensors'):
                del m._saved_tensors

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ddim_inv_steps', type=int, default=50)
    parser.add_argument('--save_feat_steps', type=int, default=50)
    parser.add_argument('--start_step', type=int, default=49)
    parser.add_argument('--ddim_eta', type=float, default=0.0)
    parser.add_argument('--H', type=int, default=512)
    parser.add_argument('--W', type=int, default=512)
    parser.add_argument('--C', type=int, default=4)
    parser.add_argument('--f', type=int, default=8)
    parser.add_argument('--T', type=float, default=1.5)
    parser.add_argument('--gamma', type=float, default=0.75)
    parser.add_argument("--attn_layer", type=str, default='6,7,8,9,10,11')
    parser.add_argument('--model_config', type=str, default='models/ldm/stable-diffusion-v1/v1-inference.yaml')
    parser.add_argument('--precomputed', type=str, default='./precomputed_feats_k')
    parser.add_argument('--ckpt', type=str, default='models/ldm/stable-diffusion-v1/model.ckpt')
    parser.add_argument('--precision', type=str, default='autocast')
    parser.add_argument('--output_path', type=str, default='output_dk')
    parser.add_argument("--without_init_adain", action='store_true')
    parser.add_argument("--without_attn_injection", action='store_true')
    parser.add_argument("--ratio", default=0.5, type=float)
    parser.add_argument("--seed", default=22, type=int)
    parser.add_argument('--data_root', type=str, default='./data_vis')
    opt = parser.parse_args()

    seed_everything(opt.seed)
    os.makedirs(opt.output_path, exist_ok=True)
    if len(opt.precomputed) > 0:
        os.makedirs(opt.precomputed, exist_ok=True)
    
    model_config = OmegaConf.load(f"{opt.model_config}")
    model = load_model_from_config(model_config, f"{opt.ckpt}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    sampler = DDIMSampler(model)
    sampler.make_schedule(ddim_num_steps=opt.save_feat_steps, ddim_eta=opt.ddim_eta, verbose=False)
    
    print("DDIM timesteps:", sampler.ddim_timesteps) 
    time_range = np.flip(sampler.ddim_timesteps)
    idx_time_dict = {t:i for i,t in enumerate(time_range)}
    time_idx_dict = {i:t for i,t in enumerate(time_range)}

    uc = model.get_learned_conditioning([""])
    shape = [opt.C, opt.H // opt.f, opt.W // opt.f]
    
    dataset = TripleImageDataset(opt.data_root, "dataset.txt", image_size=512, device=device)
    unet_model = model.model.diffusion_model
    begin = time.time()
    
    for idx in range(len(dataset)):
        print(f"Processing image {idx+1}/{len(dataset)}")
        
        # no_grad 컨텍스트로 전체 처리 감싸기
        with torch.no_grad():
            # 각 반복마다 feat_maps 초기화
            feat_maps_current = [{'config': {'gamma':opt.gamma,'T':opt.T,'cnt_k': None}} for _ in range(50)]

            def save_feature_map(fmap, fname, t):
                cur_idx = idx_time_dict[t]
                # clone().detach()로 참조 끊기
                feat_maps_current[cur_idx][fname] = fmap.clone().detach() if torch.is_tensor(fmap) else fmap

            def ddim_sampler_callback(pred_x0, xt, i):
                save_feature_map(xt, 'z_enc', i)

            cnt_img, char_img, back_img, cnt_path, char_path, back_path = dataset[idx]

            ## 마스크 적용
            for m in unet_model.modules():
                if isinstance(m, CrossAttention):
                    m.cnt_name = cnt_path

            cnt_base = os.path.splitext(os.path.basename(cnt_path))[0]
            char_base = os.path.splitext(os.path.basename(char_path))[0]
            back_base = os.path.splitext(os.path.basename(back_path))[0]

            # 변수 초기화
            char_z_enc, back_z_enc, cnt_z_enc = None, None, None
            char_feat, back_feat, cnt_feat = None, None, None

            # Style1 (char)
            char_feat_name = os.path.join(opt.precomputed, f"{char_base}_sty.pkl")
            if os.path.isfile(char_feat_name):
                with open(char_feat_name, 'rb') as h:
                    char_feat = pickle.load(h)
                    char_z_enc = torch.clone(char_feat[0]['z_enc']).detach()
            else:
                init_char = model.get_first_stage_encoding(model.encode_first_stage(char_img))
                char_z_enc, _ = sampler.encode_ddim(
                    init_char.clone(), num_steps=opt.ddim_inv_steps, unconditional_conditioning=uc,
                    end_step=time_idx_dict[opt.ddim_inv_steps-1-opt.start_step],
                    callback_ddim_timesteps=opt.save_feat_steps,
                    img_callback=ddim_sampler_callback
                )
                char_feat = copy.deepcopy(feat_maps_current)
                char_z_enc = feat_maps_current[0]['z_enc'].detach()
                with open(char_feat_name, 'wb') as h:
                    pickle.dump(char_feat, h)
                del init_char

            # Style2 (back)
            back_feat_name = os.path.join(opt.precomputed, f"{back_base}_sty.pkl")
            if os.path.isfile(back_feat_name):
                with open(back_feat_name, 'rb') as h:
                    back_feat = pickle.load(h)
                    back_z_enc = torch.clone(back_feat[0]['z_enc']).detach()
            else:
                feat_maps_current = [{'config': {'gamma':opt.gamma,'T':opt.T,'cnt_k': None}} for _ in range(50)]
                init_back = model.get_first_stage_encoding(model.encode_first_stage(back_img))
                back_z_enc, _ = sampler.encode_ddim(
                    init_back.clone(), num_steps=opt.ddim_inv_steps, unconditional_conditioning=uc,
                    end_step=time_idx_dict[opt.ddim_inv_steps-1-opt.start_step],
                    callback_ddim_timesteps=opt.save_feat_steps,
                    img_callback=ddim_sampler_callback
                )
                back_feat = copy.deepcopy(feat_maps_current)
                back_z_enc = feat_maps_current[0]['z_enc'].detach()
                with open(back_feat_name, 'wb') as h:
                    pickle.dump(back_feat, h)
                del init_back

            # Content (cnt)
            cnt_feat_name = os.path.join(opt.precomputed, f"{cnt_base}_cnt.pkl")
            if os.path.isfile(cnt_feat_name):
                with open(cnt_feat_name, 'rb') as h:
                    cnt_feat = pickle.load(h)
                    cnt_z_enc = torch.clone(cnt_feat[0]['z_enc']).detach()
            else:
                feat_maps_current = [{'config': {'gamma':opt.gamma,'T':opt.T,'cnt_k': None}} for _ in range(50)]
                init_cnt = model.get_first_stage_encoding(model.encode_first_stage(cnt_img))
                cnt_z_enc, _ = sampler.encode_ddim(
                    init_cnt.clone(), num_steps=opt.ddim_inv_steps, unconditional_conditioning=uc,
                    end_step=time_idx_dict[opt.ddim_inv_steps-1-opt.start_step],
                    callback_ddim_timesteps=opt.save_feat_steps,
                    img_callback=ddim_sampler_callback
                )
                cnt_feat = copy.deepcopy(feat_maps_current)
                cnt_z_enc = feat_maps_current[0]['z_enc'].detach()
                with open(cnt_feat_name, 'wb') as h:
                    pickle.dump(cnt_feat, h)
                del init_cnt

            ## residual injection
            def residual_injection_callback(step_idx):
                t = sampler.ddim_timesteps[step_idx]
                for block_id in range(6, 12):
                    if block_id >= len(unet_model.output_blocks):
                        break
                    for module in reversed(unet_model.output_blocks[block_id]):
                        if module.__class__.__name__.endswith("ResBlock"):
                            module.ri_timestep = int(t)
                            break

                if hasattr(model, "model_ema"):
                    ema_unet = model.model_ema.diffusion_model
                    for block_id in range(6, 12):
                        if block_id >= len(ema_unet.output_blocks):
                            break
                        for module in reversed(ema_unet.output_blocks[block_id]):
                            if module.__class__.__name__.endswith("ResBlock"):
                                module.ri_timestep = int(t)
                                break

            # High-frequency enhancement
            schedule = make_content_injection_schedule(sampler.ddim_timesteps, alpha=0.4)
            patch_decoder_resblocks_h_and_cnt_hf(unet_model, schedule, cnt_feat, ratio=opt.ratio)
            if hasattr(model, "model_ema"):
                patch_decoder_resblocks_h_and_cnt_hf(model.model_ema.diffusion_model, 
                                                        schedule, cnt_feat, ratio=opt.ratio)

            # AdaIN blending
            if opt.without_init_adain:
                adain_z_enc = cnt_z_enc.detach()
            else:
                mask = torch.tensor(np.load(os.path.join(opt.data_root, "cnt", f"{cnt_base}_mask.npy")), 
                                   dtype=torch.float32).to(device)
                mask = mask.unsqueeze(0).unsqueeze(0)
                mask = F.interpolate(mask, size=(cnt_z_enc.shape[2], cnt_z_enc.shape[3]), 
                                    mode="bilinear", align_corners=False)
                mask = mask.expand(-1, cnt_z_enc.shape[1], -1, -1)
                adain_z_enc = (mask * adain(cnt_z_enc, char_z_enc) + 
                              (1-mask) * adain(cnt_z_enc, back_z_enc)).detach()
                del mask

            # Feature merge
            merged_feat_maps = feat_merge_2sty(opt, cnt_feat, char_feat, back_feat, start_step=opt.start_step)

            # Inference
            samples_ddim, _ = sampler.sample(
                S=opt.save_feat_steps, 
                batch_size=1, 
                shape=shape, 
                verbose=False,
                unconditional_conditioning=uc, 
                eta=opt.ddim_eta, 
                x_T=adain_z_enc,
                injected_features=merged_feat_maps, 
                start_step=opt.start_step,
                callback=residual_injection_callback,
            )

            # 저장 파일명
            result_name = f"{cnt_base}_{char_base}_{back_base}.png"
            save_img_from_sample(model, samples_ddim, os.path.join(opt.output_path, result_name))

        # 명시적 메모리 정리
        del samples_ddim, char_z_enc, back_z_enc, cnt_z_enc, adain_z_enc
        del char_feat, back_feat, cnt_feat
        del merged_feat_maps, feat_maps_current
        
        # CrossAttention 캐시 클리어
        clear_attention_cache(unet_model)
        if hasattr(model, "model_ema"):
            clear_attention_cache(model.model_ema.diffusion_model)
        
        # Python 가비지 컬렉션 강제 실행
        gc.collect()
        
        # GPU 캐시 정리
        torch.cuda.empty_cache()
        
        print(f"Completed image {idx+1}/{len(dataset)}")

    print(f"Total end: {time.time() - begin}")

if __name__ == "__main__":
    main()
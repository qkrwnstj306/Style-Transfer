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

from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler

import torchvision.transforms as transforms
import torch.nn.functional as F
import time
import pickle

## 마스크 적용
from ldm.modules.attention import CrossAttention
from high_frequency_final import patch_decoder_resblocks_h_and_cnt_hf, make_content_injection_schedule

feat_maps = []

def save_img_from_sample(model, samples_ddim, fname):
    x_samples_ddim = model.decode_first_stage(samples_ddim)
    x_samples_ddim = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)
    x_samples_ddim = x_samples_ddim.cpu().permute(0, 2, 3, 1).numpy()
    x_image_torch = torch.from_numpy(x_samples_ddim).permute(0, 3, 1, 2)
    x_sample = 255. * rearrange(x_image_torch[0].cpu().numpy(), 'c h w -> h w c')
    img = Image.fromarray(x_sample.astype(np.uint8))
    img.save(fname)

def feat_merge(opt, cnt_feats, sty_feats, start_step=0):
    feat_maps = [{'config': {  
                'gamma':opt.gamma,
                'T':opt.T,
                'timestep':_,
                'cnt_k': None,
                'sty_q': None,
                }} for _ in range(50)]

    for i in range(len(feat_maps)):
        if i < (50 - start_step):
            continue
        cnt_feat = cnt_feats[i]
        sty_feat = sty_feats[i]
        ori_keys = sty_feat.keys()
        
        for ori_key in ori_keys:
            if ori_key[-1] == 'q':
                feat_maps[i][ori_key] = cnt_feat[ori_key]
            if ori_key[-1] == 'k' or ori_key[-1] == 'v':
                feat_maps[i][ori_key] = sty_feat[ori_key]
            # 같은 key에 덮어쓰지 말고 별도 key로 저장
            if ori_key[-1] == 'k':
                feat_maps[i][ori_key + '_cnt'] = cnt_feat[ori_key]
            if ori_key[-1] == 'q':
                feat_maps[i][ori_key + '_sty'] = sty_feat[ori_key]
            if ori_key[-1] == 'v':
                feat_maps[i][ori_key + '_cnt'] = cnt_feat[ori_key]
    return feat_maps

## 2가지 style feature와 content feature를 받아서 feat_merge를 수행하는 함수
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
        if i < (50 - start_step):
            continue

        cnt_feat = cnt_feats[i]
        sty_feat_1 = sty_feats_1[i]
        sty_feat_2 = sty_feats_2[i]
        ori_keys = cnt_feat.keys()  # q,k,v 다 있음

        for ori_key in ori_keys:
            # === 기본 설정: content q, style1 k/v ===
            if ori_key.endswith('q'):
                feat_maps[i][ori_key] = cnt_feat[ori_key]
            if ori_key.endswith('k') or ori_key.endswith('v'):
                feat_maps[i][ori_key] = sty_feat_1[ori_key]

            # === content 복사 ===
            if ori_key.endswith('k'):
                feat_maps[i][ori_key + '_cnt'] = cnt_feat[ori_key]
            if ori_key.endswith('v'):
                feat_maps[i][ori_key + '_cnt'] = cnt_feat[ori_key]

            # === style1 복사 ===
            if ori_key.endswith('q'):
                feat_maps[i][ori_key + '_sty1'] = sty_feat_1[ori_key]

            # === style2 복사 ===
            if ori_key.endswith('q'):
                feat_maps[i][ori_key + '_sty2'] = sty_feat_2[ori_key]
            if ori_key.endswith('k'):
                feat_maps[i][ori_key + '_sty2'] = sty_feat_2[ori_key]
            if ori_key.endswith('v'):
                feat_maps[i][ori_key + '_sty2'] = sty_feat_2[ori_key]

    return feat_maps

def load_img(path):
    image = Image.open(path).convert("RGB")
    x, y = image.size
    print(f"Loaded input image of size ({x}, {y}) from {path}")
    h = w = 512
    image = transforms.CenterCrop(min(x,y))(image)
    image = image.resize((w, h), resample=Image.Resampling.LANCZOS)
    image = np.array(image).astype(np.float32) / 255.0
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image)
    return 2.*image - 1.

def adain(cnt_feat, sty_feat):
    cnt_mean = cnt_feat.mean(dim=[0, 2, 3],keepdim=True)
    cnt_std = cnt_feat.std(dim=[0, 2, 3],keepdim=True)
    sty_mean = sty_feat.mean(dim=[0, 2, 3],keepdim=True)
    sty_std = sty_feat.std(dim=[0, 2, 3],keepdim=True)
    output = ((cnt_feat-cnt_mean)/cnt_std)*sty_std + sty_mean
    return output

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
    parser.add_argument('--cnt', default = './data/cnt')
    parser.add_argument('--sty', default = './data/sty')
    parser.add_argument('--ddim_inv_steps', type=int, default=50, help='DDIM eta')
    parser.add_argument('--save_feat_steps', type=int, default=50, help='DDIM eta')
    parser.add_argument('--start_step', type=int, default=49, help='DDIM eta')
    parser.add_argument('--ddim_eta', type=float, default=0.0, help='DDIM eta')
    parser.add_argument('--H', type=int, default=512, help='image height, in pixel space')
    parser.add_argument('--W', type=int, default=512, help='image width, in pixel space')
    parser.add_argument('--C', type=int, default=4, help='latent channels')
    parser.add_argument('--f', type=int, default=8, help='downsampling factor')
    parser.add_argument('--T', type=float, default=1.5, help='attention temperature scaling hyperparameter')
    parser.add_argument('--gamma', type=float, default=0.75, help='query preservation hyperparameter')
    parser.add_argument("--attn_layer", type=str, default='6,7,8,9,10,11', help='injection attention feature layers')
    parser.add_argument('--model_config', type=str, default='models/ldm/stable-diffusion-v1/v1-inference.yaml', help='model config')
    parser.add_argument('--precomputed', type=str, default='./precomputed_feats_k', help='save path for precomputed feature')
    parser.add_argument('--ckpt', type=str, default='models/ldm/stable-diffusion-v1/model.ckpt', help='model checkpoint')
    parser.add_argument('--precision', type=str, default='autocast', help='choices: ["full", "autocast"]')
    parser.add_argument('--output_path', type=str, default='output_dk')
    parser.add_argument("--without_init_adain", action='store_true')
    parser.add_argument("--without_attn_injection", action='store_true')
    parser.add_argument("--ratio", default=0.5, type=float, help="content high-freq ratio")
    opt = parser.parse_args()

    feat_path_root = opt.precomputed

    seed_everything(22)
    output_path = opt.output_path
    os.makedirs(output_path, exist_ok=True)
    if len(feat_path_root) > 0:
        os.makedirs(feat_path_root, exist_ok=True)
    
    model_config = OmegaConf.load(f"{opt.model_config}")
    model = load_model_from_config(model_config, f"{opt.ckpt}")
   
    
    self_attn_output_block_indices = list(map(int, opt.attn_layer.split(',')))
    ddim_inversion_steps = opt.ddim_inv_steps
    save_feature_timesteps = ddim_steps = opt.save_feat_steps

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = model.to(device)
    unet_model = model.model.diffusion_model
    sampler = DDIMSampler(model)
    sampler.make_schedule(ddim_num_steps=ddim_steps, ddim_eta=opt.ddim_eta, verbose=False)
    
    # #### q,k, sim 저장용 ##### 
    # # injection 대상이 되는 layer
    # target_layer = 11          # e.g. 11
    # # sampler.ddim_timesteps 에서 timestep
    # target_t_list   = [4, 14, 24, 34, 44]   

    
    
    print("DDIM timesteps:", sampler.ddim_timesteps) 
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
                'gamma':opt.gamma,
                'T':opt.T,
                'cnt_k': None,
                }} for _ in range(50)]

    def ddim_sampler_callback(pred_x0, xt, i):
    
        save_feature_maps_callback(i)
        save_feature_map(xt, 'z_enc', i)
        
    # def sampling_callback(i):
    #     t = sampler.ddim_timesteps[i]
    #     print(f"[SamplerCallback] step index={i}, timestep={t}")
        
        
    def save_feature_maps(blocks, i, feature_type="input_block"):
        block_idx = 0
        for block_idx, block in enumerate(blocks):
            if len(block) > 1 and "SpatialTransformer" in str(type(block[1])):
                if block_idx in self_attn_output_block_indices:
                    print(f"block_idx = {block_idx}")
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

    def save_feature_map(feature_map, filename, time):
        global feat_maps
        cur_idx = idx_time_dict[time]
        feat_maps[cur_idx][f"{filename}"] = feature_map
        
    ## residual injection
    # 매 DDIM 스텝마다 timestep을 hook 모듈에 설정하는 콜백
    def residual_injection_callback(step_idx):
        t = sampler.ddim_timesteps[step_idx]
        
        # residual_high 저장 경로에서 불러오기
        # 예: precomputed_feats/.../{image_name}_residuals_all.pkl
        # 현재 content 이미지의 high-freq residual 불러오기

        for block_id in range(6, 12):
            if block_id >= len(unet_model.output_blocks):
                break

            for module in reversed(unet_model.output_blocks[block_id]):
                if module.__class__.__name__.endswith("ResBlock"):
                    module.ri_timestep = int(t)
                    break
                # EMA 모델도 동일하게

        if hasattr(model, "model_ema"):
            ema_unet = model.model_ema.diffusion_model
            for block_id in range(6, 12):
                if block_id >= len(ema_unet.output_blocks):
                    break
                for module in reversed(ema_unet.output_blocks[block_id]):
                    if module.__class__.__name__.endswith("ResBlock"):
                        module.ri_timestep = int(t)
                        break

    start_step = opt.start_step
    precision_scope = autocast if opt.precision=="autocast" else nullcontext
    uc = model.get_learned_conditioning([""])
    shape = [opt.C, opt.H // opt.f, opt.W // opt.f]
    
    ## 마스크 적용
    # 허용할 이미지 확장자
    IMG_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp")
    
    sty_img_list = sorted([
        f for f in os.listdir(opt.sty)
        if f.lower().endswith(IMG_EXTENSIONS)
        ])
    
    cnt_img_list = sorted([
        f for f in os.listdir(opt.cnt)
        if f.lower().endswith(IMG_EXTENSIONS)
    ])

    begin = time.time()
    """Single-Style Transfer"""
    # for sty_name in sty_img_list:
    #     sty_name_ = os.path.join(opt.sty, sty_name)
    #     init_sty = load_img(sty_name_).to(device)
    #     seed = -1
    #     sty_feat_name = os.path.join(feat_path_root, os.path.basename(sty_name).split('.')[0] + '_sty.pkl')
    #     sty_z_enc = None

    #     if len(feat_path_root) > 0 and os.path.isfile(sty_feat_name):
    #         print("Precomputed style feature loading: ", sty_feat_name)
    #         with open(sty_feat_name, 'rb') as h:
    #             sty_feat = pickle.load(h)
    #             sty_z_enc = torch.clone(sty_feat[0]['z_enc'])
    #     else:
    #         init_sty = model.get_first_stage_encoding(model.encode_first_stage(init_sty))
    #         sty_z_enc, _ = sampler.encode_ddim(init_sty.clone(), num_steps=ddim_inversion_steps, unconditional_conditioning=uc, \
    #                                             end_step=time_idx_dict[ddim_inversion_steps-1-start_step], \
    #                                             callback_ddim_timesteps=save_feature_timesteps,
    #                                             img_callback=ddim_sampler_callback)
    #         sty_feat = copy.deepcopy(feat_maps)
    #         sty_z_enc = feat_maps[0]['z_enc']

    """Multi-Style Transfer"""
    sty_feats = []
    sty_z_enc_list = []
    for idx, sty_name in enumerate(sty_img_list):
        sty_name_ = os.path.join(opt.sty, sty_name)
        init_sty = load_img(sty_name_).to(device)

        seed = -1
        sty_feat_name = os.path.join(feat_path_root, os.path.basename(sty_name).split('.')[0] + '_sty.pkl')
        sty_z_enc = None

        if len(feat_path_root) > 0 and os.path.isfile(sty_feat_name):
            print("Precomputed style feature loading: ", sty_feat_name)
            with open(sty_feat_name, 'rb') as h:
                sty_feat = pickle.load(h)
                sty_z_enc = torch.clone(sty_feat[0]['z_enc'])
                #style의 pkl 파일에 저장된 feat_maps를 불러와 sty1,2를 저장
                sty_feats.append(sty_feat)# sty의 feat_maps를 저장
                sty_z_enc_list.append(sty_z_enc)# sty의 z_enc를 저장
        else:
            init_sty = model.get_first_stage_encoding(model.encode_first_stage(init_sty))
            sty_z_enc, _ = sampler.encode_ddim(init_sty.clone(), num_steps=ddim_inversion_steps, unconditional_conditioning=uc, \
                                                end_step=time_idx_dict[ddim_inversion_steps-1-start_step], \
                                                callback_ddim_timesteps=save_feature_timesteps,
                                                img_callback=ddim_sampler_callback)
            sty_feat = copy.deepcopy(feat_maps)
            sty_z_enc = feat_maps[0]['z_enc']

            
    sty_feat_1, sty_feat_2 = sty_feats
    sty_z_enc_1, sty_z_enc_2 = sty_z_enc_list

    for cnt_name in cnt_img_list:
        cnt_name_ = os.path.join(opt.cnt, cnt_name)
        init_cnt = load_img(cnt_name_).to(device)
        cnt_feat_name = os.path.join(feat_path_root, os.path.basename(cnt_name).split('.')[0] + '_cnt.pkl')
        cnt_feat = None
        
        # ────── 여기서 residual injection 준비 ──────
            
        # content image의 확장자를 자동으로 감지하여 residual path 생성
        cnt_base_name = os.path.basename(cnt_name)

        cnt_no_ext, _ = os.path.splitext(cnt_base_name)
        # residuals_all 파일명은 "avril_residuals_all.pkl" 로
        residual_path = os.path.join(feat_path_root, f"{cnt_no_ext}_residuals_all.pkl")


        #callback에서 사용할 수 있도록 global 변수에 설정
        global current_cnt_residuals_all_path
        current_cnt_residuals_all_path = residual_path

        if os.path.exists(residual_path):
            with open(residual_path, 'rb') as f:
                residuals_all = pickle.load(f)
            print(f"[ResidualInjection] Loaded residuals from {residual_path}")
        else:
            residuals_all = None
            print(f"[ResidualInjection] No residuals file at {residual_path}")

        schedule = make_content_injection_schedule(sampler.ddim_timesteps, alpha=0.4)

        patch_decoder_resblocks_h_and_cnt_hf(unet_model, 
                                                schedule, 
                                                residuals_all, ratio=opt.ratio)
        if hasattr(model, "model_ema"):
            patch_decoder_resblocks_h_and_cnt_hf(model.model_ema.diffusion_model, 
                                                    schedule, 
                                                    residuals_all, ratio=opt.ratio)
        # # ────────────────────────────────────────────

        # ddim inversion encoding
        if len(feat_path_root) > 0 and os.path.isfile(cnt_feat_name):
            print("Precomputed content feature loading: ", cnt_feat_name)
            with open(cnt_feat_name, 'rb') as h:
                cnt_feat = pickle.load(h)
                cnt_z_enc = torch.clone(cnt_feat[0]['z_enc'])
        else:
            init_cnt = model.get_first_stage_encoding(model.encode_first_stage(init_cnt))
            cnt_z_enc, _ = sampler.encode_ddim(init_cnt.clone(), num_steps=ddim_inversion_steps, unconditional_conditioning=uc, \
                                                end_step=time_idx_dict[ddim_inversion_steps-1-start_step], \
                                                callback_ddim_timesteps=save_feature_timesteps,
                                                img_callback=ddim_sampler_callback)
            cnt_feat = copy.deepcopy(feat_maps)
            cnt_z_enc = feat_maps[0]['z_enc']

        with torch.no_grad():
            with precision_scope("cuda"):
                with model.ema_scope():
                    output_name = f"{os.path.basename(cnt_name).split('.')[0]}_stylized_{os.path.basename(sty_name).split('.')[0]}.png"
                    
                    print(f"Inversion end: {time.time() - begin}")
                    
                    if opt.without_init_adain:
                        adain_z_enc = cnt_z_enc
                    else:
                        # adain_z_enc = adain(cnt_z_enc, sty_z_enc)
                        mask = torch.tensor(np.load(os.path.join(opt.cnt, f"{os.path.basename(cnt_name).split('.')[0]}_mask.npy")), dtype=torch.float32).to(device)
                        mask = mask.unsqueeze(0).unsqueeze(0)
                        mask = F.interpolate(mask, size=(cnt_z_enc.shape[2], cnt_z_enc.shape[3]), mode="bilinear", align_corners=False)
                        mask = mask.expand(-1, cnt_z_enc.shape[1], -1, -1)
                        adain_z_enc = mask * adain(cnt_z_enc, sty_z_enc_1) + (1-mask) * adain(cnt_z_enc, sty_z_enc_2)

                    feat_maps = feat_merge_2sty(opt, cnt_feat, sty_feat_1, sty_feat_2, start_step=start_step)
                    if opt.without_attn_injection:
                        feat_maps = None
                        
                    print(f"{sty_name_}, {cnt_name_}")
                    
                    ## 마스크 적용
                    for m in unet_model.modules():
                        if isinstance(m, CrossAttention):
                            m.sty_name = sty_name_
                            m.cnt_name = cnt_name_
                    
                    # # ─── q, k, sim 저장용 ───
                    # # style/content 이름을 파일명에 반영하고, 매번 리셋
                    # target_block = unet_model.output_blocks[target_layer]
                    # for module in target_block:
                    #     if isinstance(module, SpatialTransformer):
                    #         module.transformer_blocks[0].attn1.target_t_list =target_t_list
                    #         module.transformer_blocks[0].attn1.cnt_name=cnt_name_
                    #         module.transformer_blocks[0].attn1.sty_name=sty_name_
                    #         module.transformer_blocks[0].attn1.layer_id=target_layer
                            
                    
                    # inference
                    samples_ddim, intermediates = sampler.sample(
                        S=ddim_steps,
                        batch_size=1,
                        shape=shape,
                        verbose=False,
                        unconditional_conditioning=uc,
                        eta=opt.ddim_eta,
                        x_T=adain_z_enc,
                        injected_features=feat_maps,
                        start_step=start_step,
                        # ## 마스크 적용
                        sty_name=sty_name_,
                        cnt_name=cnt_name_,
                        # # ## residual injection
                        callback=residual_injection_callback,
                        ### q, k, sim 저장
                        injection_config={'timestep': t}
                    )

                    x_samples_ddim = model.decode_first_stage(samples_ddim)
                    x_samples_ddim = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)
                    x_samples_ddim = x_samples_ddim.cpu().permute(0, 2, 3, 1).numpy()
                    x_image_torch = torch.from_numpy(x_samples_ddim).permute(0, 3, 1, 2)
                    x_sample = 255. * rearrange(x_image_torch[0].cpu().numpy(), 'c h w -> h w c')
                    img = Image.fromarray(x_sample.astype(np.uint8))

                    img.save(os.path.join(output_path, output_name))
                    
                if len(feat_path_root) > 0:
                    print("Save features")
                    if not os.path.isfile(cnt_feat_name):
                        with open(cnt_feat_name, 'wb') as h:
                            pickle.dump(cnt_feat, h)
                    if not os.path.isfile(sty_feat_name):
                        with open(sty_feat_name, 'wb') as h:
                            pickle.dump(sty_feat, h)

    print(f"Total end: {time.time() - begin}")

if __name__ == "__main__":
    main()
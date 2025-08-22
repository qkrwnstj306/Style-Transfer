
## 백업용

from inspect import isfunction
import math
import torch
import torch.nn.functional as F
from torch import nn, einsum
from einops import rearrange, repeat
import pickle
import os

from ldm.modules.diffusionmodules.util import checkpoint

## 마스크 적용
from math import sqrt
import numpy as np


def exists(val):
    return val is not None


def uniq(arr):
    return{el: True for el in arr}.keys()


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


def max_neg_value(t):
    return -torch.finfo(t.dtype).max


def init_(tensor):
    dim = tensor.shape[-1]
    std = 1 / math.sqrt(dim)
    tensor.uniform_(-std, std)
    return tensor


# feedforward
class GEGLU(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out * 2)

    def forward(self, x):
        x, gate = self.proj(x).chunk(2, dim=-1)
        return x * F.gelu(gate)


class FeedForward(nn.Module):
    def __init__(self, dim, dim_out=None, mult=4, glu=False, dropout=0.):
        super().__init__()
        inner_dim = int(dim * mult)
        dim_out = default(dim_out, dim)
        project_in = nn.Sequential(
            nn.Linear(dim, inner_dim),
            nn.GELU()
        ) if not glu else GEGLU(dim, inner_dim)

        self.net = nn.Sequential(
            project_in,
            nn.Dropout(dropout),
            nn.Linear(inner_dim, dim_out)
        )

    def forward(self, x):
        return self.net(x)


def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module


def Normalize(in_channels):
    return torch.nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)


class LinearAttention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias = False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x)
        q, k, v = rearrange(qkv, 'b (qkv heads c) h w -> qkv b heads c (h w)', heads = self.heads, qkv=3)
        k = k.softmax(dim=-1)  
        context = torch.einsum('bhdn,bhen->bhde', k, v)
        out = torch.einsum('bhde,bhdn->bhen', context, q)
        out = rearrange(out, 'b heads c (h w) -> b (heads c) h w', heads=self.heads, h=h, w=w)
        return self.to_out(out)


# class SpatialSelfAttention(nn.Module):
#     def __init__(self, in_channels):
#         super().__init__()
#         self.in_channels = in_channels

#         self.norm = Normalize(in_channels)
#         self.q = torch.nn.Conv2d(in_channels,
#                                  in_channels,
#                                  kernel_size=1,
#                                  stride=1,
#                                  padding=0)
#         self.k = torch.nn.Conv2d(in_channels,
#                                  in_channels,
#                                  kernel_size=1,
#                                  stride=1,
#                                  padding=0)
#         self.v = torch.nn.Conv2d(in_channels,
#                                  in_channels,
#                                  kernel_size=1,
#                                  stride=1,
#                                  padding=0)
#         self.proj_out = torch.nn.Conv2d(in_channels,
#                                         in_channels,
#                                         kernel_size=1,
#                                         stride=1,
#                                         padding=0)

#     def forward(self, x):
#         h_ = x
#         h_ = self.norm(h_)
#         q = self.q(h_)
#         k = self.k(h_)
#         v = self.v(h_)

#         # compute attention
#         b,c,h,w = q.shape
#         q = rearrange(q, 'b c h w -> b (h w) c')
#         k = rearrange(k, 'b c h w -> b c (h w)')
#         w_ = torch.einsum('bij,bjk->bik', q, k)

#         w_ = w_ * (int(c)**(-0.5))
#         w_ = torch.nn.functional.softmax(w_, dim=2)

#         # attend to values
#         v = rearrange(v, 'b c h w -> b c (h w)')
#         w_ = rearrange(w_, 'b i j -> b j i')
#         h_ = torch.einsum('bij,bjk->bik', v, w_)
#         h_ = rearrange(h_, 'b c (h w) -> b c h w', h=h)
#         h_ = self.proj_out(h_)

#         return x+h_


class CrossAttention(nn.Module):
    def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        # print(f"context_dim is exists: {exists(context_dim)}")
        context_dim = default(context_dim, query_dim)
        
        self.scale = dim_head ** -0.5
      
        self.heads = heads

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)
        
        
        
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, query_dim),
            nn.Dropout(dropout)
        )

        self.attn = None
        self.q = None
        self.k = None
        self.v = None
        self.qk_sim = None
        
        ## 마스크 적용
        self.sty_name = None
        self.cnt_name = None
        
        ## q, k, sim 저장용
        self.target_t_list = None
        self.layer_id = None
    
    
    def get_batch_sim(self, q, k, num_heads, **kwargs):
        
        q = rearrange(q, "(b h) n d -> h (b n) d", h=num_heads)
        k = rearrange(k, "(b h) n d -> h (b n) d", h=num_heads)
        
        sim = torch.einsum("h i d, h j d -> h i j", q, k) * self.scale
        return sim 
    
    
    def get_batch_sim_with_mask(self, cc_sim, delta_q, delta_k, q, k, num_heads, sty_name, cnt_name, mask_path=None, attn_matrix_scale=1.0, injection_config=None,target_t_list=None):
      
        q = rearrange(q, "(b h) n d -> h (b n) d", h=num_heads)
        k = rearrange(k, "(b h) n d -> h (b n) d", h=num_heads)

        sty_name = sty_name
        cnt_name = cnt_name

        sim = torch.einsum("h i d, h j d -> h i j", q, k)
        
        sim *= attn_matrix_scale
        sim *= self.scale


        head_num = sim.shape[0]
        pixel_size = sim.shape[1]
        h = w = int(sqrt(pixel_size))

        ##### z* 버전    
        # 기존 mask 적용 방식
        sim_reshaped = sim.reshape(head_num, h, w, pixel_size)
        ## cc_sim은 content-content query-key 내적에 scale까지 한 값.
        cc_sim_reshaped = cc_sim.reshape(head_num, h, w, pixel_size)
        
        delta_q = rearrange(delta_q, "(b h) n d -> h (b n) d", h=num_heads)
        delta_k = rearrange(delta_k, "(b h) n d -> h (b n) d", h=num_heads)
        max_sim = torch.einsum("h i d, h j d -> h i j", delta_q, delta_k)
        max_sim_reshaped = max_sim.reshape(head_num, h, w, pixel_size)

        min_cc_sim_reshaped, _ = torch.min(
            cc_sim_reshaped, dim=3, keepdim=True)
        max_sim_reshaped, _ = torch.max(max_sim_reshaped, dim=3, keepdim=True)
        start = 0.5
        end = -0.5
        length = w


        # # if mask_path is not None and sty_name == "c_style.jpg":
        # #     mask = torch.tensor(np.load(mask_path), dtype=torch.float32).cuda()
        # #     mask[mask < 0.5] = -1.0
        # #     mask[mask > 0.5] = 1.0
        # #     mask *= -1.0
        
        # # # # 수정본, 배경에 마스크 적용
        # # elif mask_path is not None and sty_name == "b_style.jpg":
        # #     mask = torch.tensor(np.load(mask_path), dtype=torch.float32).cuda()
        # #     mask = 1.0 - mask   # 배경용 반전
        # #     mask[mask < 0.5] = -1.0
        # #     mask[mask > 0.5] = 1.0
        # #     mask *= -1.0
            
            
        # # else:
        # #     print("ERROR: mask npy not found!!!")
        # #     mask = torch.tensor([[1., 1., 1., 1.],
        # #                         [1., -1., -1., 1.],
        # #                         [1., -1., -1., 1.],
        # #                         [-1., -1., -1., -1.]], dtype=torch.float32).cuda()
        
        
        mask = torch.tensor(np.load(mask_path), dtype=torch.float32).cuda()
        mask[mask < 0.5] = -1.0
        mask[mask > 0.5] = 1.0
        mask *= -1.0
        
    
        mask = mask.unsqueeze(0).unsqueeze(0)
        
        
        
        ### zero_star에서 사용한 방식
        mask = F.interpolate(mask, size=(
            h, w), mode='bilinear', align_corners=False)
        mask = mask.reshape(1, h, w, 1).to(sim.device)  # (1, h, w, 1)
        gradual_vanished_array = mask.reshape(1, h, w, 1).to(sim.device)
        delta = min_cc_sim_reshaped - max_sim_reshaped
        gradual_vanished_mask = (delta)[:, :, :, :] * gradual_vanished_array
        print(f"gradual_vanished_mask shape: {gradual_vanished_mask.shape}")
        sim_reshaped[:, :length, :, :] += gradual_vanished_mask
        
        sim = sim_reshaped.reshape(head_num, pixel_size, pixel_size)
        
        
        ### 여기선 alpha가 더해진 어텐션 맵 자체를 저장
        if isinstance(injection_config, dict) and target_t_list is not None:
            t = injection_config.get('timestep')
            
            # self 객체에서 target_t(리스트)와 target_layer(정수)를 가져옴
            current_layer_id = getattr(self, 'layer_id', None)
            # 현재 timestep이 target_t 리스트에 있을 때만 저장.
            if t in target_t_list:
                save_dir = 'saved_sim'
                os.makedirs(save_dir, exist_ok=True)
                
                # 파일명에 layer, timestep, 스타일, 콘텐츠 이름 포함
                fn = f"layer{current_layer_id}_t{t+1}_{os.path.basename(self.sty_name)}_{os.path.basename(self.cnt_name)}_sim.pkl"
                path = os.path.join(save_dir, fn)
                
                with open(path, 'wb') as f:
                    pickle.dump(sim.detach().cpu(), f)
                
                print(f"[SimCapture] saved sim map -> {path}")
        # ------------------- 추가/수정된 코드 종료 -------------------
        
        ### z* 버전
        
        ##### delta 수정 버전.
        # mask = torch.tensor(np.load(mask_path), dtype=torch.float32).cuda()
        # # mask[mask < 0.5] = -1.0
        # # mask[mask > 0.5] = 1.0
        # # mask *= -1.0
        # diff_mask = mask
        # diff_mask[diff_mask < 0.5] = -1.0
        # diff_mask[diff_mask > 0.5] = 1.0
    
        # mask = mask.unsqueeze(0).unsqueeze(0)
        # diff_mask = diff_mask.unsqueeze(0).unsqueeze(0)
        
        # mask = F.interpolate(mask, size=(
        #     h, w), mode='bilinear', align_corners=False)
        # diff_mask = F.interpolate(diff_mask, size=(
        #     h, w), mode='bilinear', align_corners=False)
        # mask = mask.reshape(1, h, w, 1).to(sim.device)  # (1, h, w, 1)
        # diff_mask = diff_mask.reshape(1, h, w, 1).to(sim.device)
        
        # masked_cc_sim_reshaped = cc_sim_reshaped * mask  # (h, w, pixel_size)

        # masked_cc_sim_mu = masked_cc_sim_reshaped.mean(dim=(1,2,3), keepdim=True)  # (H, h, w, 1)
        # max_sim_reshaped_mu = max_sim_reshaped.mean(dim=(1,2,3), keepdim=True)  # (H, h, w, 1)
        
        # diff = (masked_cc_sim_mu - max_sim_reshaped_mu).abs()  # (h, w, 1)

        
        
        # gradual_vanished_mask = diff[:, :, :, :]  * diff_mask # (h, w, 1)
        # sim_reshaped[:, :length, :, :] += gradual_vanished_mask
        # sim = sim_reshaped.reshape(head_num, pixel_size, pixel_size)
        ##### delta 수정 버전.
        
        
        return sim
    
    
    def forward(self,
                x,
                context=None,
                mask=None,
                q_injected=None, 
                k_injected=None, 
                v_injected=None, 
                cnt_k_injected=None,
                sty_q_injected=None,
                cnt_v_injected=None,
                injection_config=None,):
        self.attn = None
        batch, seq_len, _ = x.shape
        h = self.heads
        b = x.shape[0]
     
        attn_matrix_scale = 1.0
        q_mix = 0.
        is_cross = context is not None
        

        
        # import builtins
        if injection_config is not None:
            # cfg = builtins.feat_maps[builtins.global_step_idx]['config']
            # attn_matrix_scale = cfg.get("T", 1.0)#injection_config['T']
            #q_mix = cfg.get("gamma", 0.0)#injection_config['gamma']
            
            ## 원본
            attn_matrix_scale = injection_config['T']
            q_mix = injection_config['gamma']
            
        

        if q_injected is None:
            q = self.to_q(x)
            q = rearrange(q, 'b n (h d) -> (b h) n d', h=h)
        
        else:
            q_uncond = q_injected
            q_in = torch.cat([q_uncond]*b)
            q_ = self.to_q(x)
            q_ = rearrange(q_, 'b n (h d) -> (b h) n d', h=h)
            
            # q = q_in
            q = q_in * 0.5 + (1. - 0.5) * q_ #content query가 q_in이다.
           
            
        context = default(context, x)

        if k_injected is None:
            k = self.to_k(context)
            k = rearrange(k, 'b m (h d) -> (b h) m d', h=h)
            
        else:
            k_uncond = k_injected
            k = torch.cat([k_uncond]*b ,dim=0)
           

            
        if v_injected is None:
            v = self.to_v(context)
            v = rearrange(v, 'b m (h d) -> (b h) m d', h=h)
         
        else:
            v_uncond = v_injected
            v = torch.cat([v_uncond]*b ,dim=0)
           

        self.q = q
        self.k = k
        self.v = v
        
       
        
        
        ##################### 마스크 적용 시작 ######################
        mask_path = self.cnt_name.replace(".jpg", "_mask.npy")
        is_mask_exists = os.path.exists(mask_path)  # mask_path가 존재하는지 확인
        
        use_mask = (
            self.sty_name is not None
            and self.cnt_name is not None
            and not is_cross
            and is_mask_exists
        )
        


        if use_mask:
            self.sty_name = os.path.basename(self.sty_name)
            if q_injected is not None and k_injected is not None:
                # ## ver 1. Q^cs StyleID 그대로 사용
                # q_cnt = copy.deepcopy(self.q)
                # ver 2. z*처럼 q_cnt는 only content query
                q_cnt = q_in

                k_cnt = torch.cat([cnt_k_injected]*b, dim=0)
                q_sty = torch.cat([sty_q_injected]*b, dim=0)
                v_cnt = torch.cat([cnt_v_injected]*b, dim=0)
                
                cc_sim = self.get_batch_sim(
                    q=q_cnt,
                    k=k_cnt,
                    num_heads=h,
                )
                

                ### self.q 저장
                # 내가 지정한 layer·내가 지정한 timestep 에서만 q,k 저장
                # 1) Q/K 캡처용 플래그 + 유효한 dict일 때만 get() 호출
                if isinstance(injection_config, dict) and self.target_t_list is not None:
                    t = injection_config.get('timestep')
                    if t in self.target_t_list:
                        # 1) raw q/k 뽑아오기
                        raw_q = self.to_q(x).detach().cpu()         # (1, N, heads*d)
                        raw_k = self.to_k(x).detach().cpu()  # (1, N, heads*d)

                        # 2) (1, N, heads*d) → (1, N, heads, d) → (heads, N, d)
                        B, N, C = raw_q.shape
                        heads = self.heads
                        d     = C // heads
                        q     = raw_q.view(B, N, heads, d).permute(0,2,1,3).squeeze(0)  # (heads, N, d)
                        k     = raw_k.view(B, N, heads, d).permute(0,2,1,3).squeeze(0)  # (heads, N, d)

                        # 3) 원하는 키 이름
                        q_key = f"output_block_{self.layer_id}_self_attn_q"
                        k_key = f"output_block_{self.layer_id}_self_attn_k"

                        # 4) 리스트-of-dict 포맷으로 패키징
                        feat_list = [{ q_key: q, k_key: k }]

                        # 5) 디스크에 저장
                        save_dir = 'saved_qk'
                        os.makedirs(save_dir, exist_ok=True)
                        cnt_basename = os.path.basename(self.cnt_name)
                        fn   = f"layer{self.layer_id}_t{t+1}_{self.sty_name}_{cnt_basename}_qk.pkl"
                        path = os.path.join(save_dir, fn)
                        with open(path, 'wb') as f:
                            pickle.dump(feat_list, f)
                        print(f"[QKCapture] saved q,k → {path}")
                        # ───────────────────────────────────────────────────────────
                
                
                sim = self.get_batch_sim_with_mask(
                    cc_sim=cc_sim,
                    q=self.q, ##  self.q 면 q_cs를 의미, q_cnt면 감마가 적용되지않은 cnt그대로
                    delta_q=q_cnt,
                    delta_k=self.k, # self.k = k_sty와 같음. inject당시 sty에서 key와 value를 가져오기 때문.
                    k=self.k,
                    num_heads=h,
                    sty_name=self.sty_name,
                    cnt_name=self.cnt_name,
                    mask_path=mask_path,
                    attn_matrix_scale=attn_matrix_scale,
                    injection_config=injection_config,
                    target_t_list=self.target_t_list,
                )
                
                h, BN, _ = sim.shape
                B = b
                N = BN // B
                # 1) (h, B*N, B*N) -> (h, B, N, B, N)
                sim = sim.view(h, B, N, B, N)
                # 2) (h, B, N, B, N) -> (B, h, N, B, N)
                sim = sim.permute(1, 0, 2, 3, 4)
                sim = sim.reshape(B*h, N, N)
                
                attn = sim.softmax(dim=-1)
               
                self.attn = attn
                out = einsum('b i j, b j d -> b i d', attn, v)
                
                
                cc_sim  = cc_sim.view(h, B, N, B, N)
                cc_sim = cc_sim.permute(1, 0, 2, 3, 4)
                cc_sim = cc_sim.reshape(B*h, N, N)
                
            
                
                
                
                # # 오메가 마스크 적용 ###################
                # if mask_path is not None and self.sty_name == "c_style.jpg":
                #     mask = torch.tensor(np.load(mask_path), dtype=torch.float32).cuda()
                    
                # # # 수정본, 배경에 마스크 적용
                # elif mask_path is not None and self.sty_name == "b_style.jpg":
                #     mask = torch.tensor(np.load(mask_path), dtype=torch.float32).cuda()
                #     mask = 1.0 - mask   # 배경용 반전
                    
                # else:
                #     print("ERROR: mask npy not found!!!")
                #     mask = torch.tensor([[1., 1., 1., 1.],
                #                         [1., -1., -1., 1.],
                #                         [1., -1., -1., 1.],
                #                         [-1., -1., -1., -1.]], dtype=torch.float32).cuda()
                
                # mask = mask.unsqueeze(0).unsqueeze(0)
                # H, N, D = out.shape 
                # mask = F.interpolate(mask, size=(int(math.sqrt(out.shape[1])),
                #                      int(math.sqrt(out.shape[1]))),
                #          mode='bilinear', align_corners=False)
                
                # mask = mask.view(1, N, 1)              # (1, N, 1)
                # mask = mask.expand(H, -1, -1)          # (H, N, 1)

                # # style/content blend (head별 동일 mask 적용)
                # attn_cnt = cc_sim.softmax(dim=-1)
                # out_cnt = einsum('b i j, b j d -> b i d', attn_cnt, v_cnt)
                # # out = mask * out + (1-mask) * out_cnt
                # # cat 아닐 때,
                # out = rearrange(out, '(b h) n d -> b n (h d)', h=h)
                # # 오메가 마스크 적용 ###################
                
                
                ### z* 방식 ############
                cat_sim = torch.cat((sim, cc_sim), 2)
                cat_v = torch.cat((v, v_cnt), 1)
                
                cat_sim = cat_sim.softmax(-1)
                
                cat_out = einsum('b i j, b j d -> b i d', cat_sim, cat_v)
                # cat시에는
                out = rearrange(cat_out, 'h (b n) d -> b n (h d)', h=h, b=b)
                ### z* 방식 ############
                
                

                
                
                
            # style injection이 일어나지 않는 경우 -- 원본과 동일하게 진행
            else:
                sim = einsum('b i d, b j d -> b i j', q, k)
                sim *= attn_matrix_scale
                sim *= self.scale
                attn = sim.softmax(dim=-1)
                self.attn = attn
                
                
                out = einsum('b i j, b j d -> b i d', attn, v)
            
                out = rearrange(out, '(b h) n d -> b n (h d)', h=h)
                
        # {마스크용 npy파일이 없으면 마스킹 적용 x} or {self.attn2 즉, cross-attention}  -- 원본과 동일하게 진행  
        else:
            sim = einsum('b i d, b j d -> b i j', q, k)
            if q_injected is not None or k_injected is not None:
            # print(attn_matrix_scale, 'attn_matrix_scale')
                sim *= attn_matrix_scale    
            sim *= self.scale
            attn = sim.softmax(dim=-1)
            self.attn = attn

            out = einsum('b i j, b j d -> b i d', attn, v)
            out = rearrange(out, '(b h) n d -> b n (h d)', h=h)
            
            # print(f"마스크 적용 안함, sim.shape: {sim.shape} \n")
        ################# 마스크 적용 끝 ###################
        
        # ################## 원본 ###################
        # sim = einsum('b i d, b j d -> b i j', q, k)
        
        
        # sim *= self.scale
        # self.qk_sim = sim

        # if exists(mask):
        #     mask = rearrange(mask, 'b ... -> b (...)')
        #     max_neg_value = -torch.finfo(sim.dtype).max
        #     mask = repeat(mask, 'b j -> (b h) () j', h=h)
        #     sim.masked_fill_(~mask, max_neg_value)

        # # attention, what we cannot get enough of
        # attn = sim.softmax(dim=-1)
        # self.attn = attn

        # out = einsum('b i j, b j d -> b i d', attn, v)
        # out = rearrange(out, '(b h) n d -> b n (h d)', h=h)
        # ################## 원본 끝 ###################
        
        
        return self.to_out(out)


class BasicTransformerBlock(nn.Module):
    def __init__(self, dim, n_heads, d_head, dropout=0., context_dim=None, gated_ff=True, checkpoint=True):
        super().__init__()
        self.attn1 = CrossAttention(query_dim=dim, heads=n_heads, dim_head=d_head, dropout=dropout)  # is a self-attention
        self.ff = FeedForward(dim, dropout=dropout, glu=gated_ff)
        self.attn2 = CrossAttention(query_dim=dim, context_dim=context_dim,
                                    heads=n_heads, dim_head=d_head, dropout=dropout)  # is self-attn if context is none
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)
        self.checkpoint = checkpoint
        
    def forward(self,
                x,
                context=None,
                self_attn_q_injected=None,
                self_attn_k_injected=None,
                self_attn_v_injected=None,
                self_attn_cnt_k_injected=None,
                self_attn_sty_q_injected=None,
                self_attn_cnt_v_injected=None,
                injection_config=None,
                ):
        return checkpoint(self._forward, (x,
                                          context,
                                          self_attn_q_injected,
                                          self_attn_k_injected,
                                          self_attn_v_injected,
                                          self_attn_cnt_k_injected,
                                          self_attn_sty_q_injected,
                                          self_attn_cnt_v_injected,
                                          injection_config,), self.parameters(), self.checkpoint)

    def _forward(self,
                 x,
                 context=None,
                 self_attn_q_injected=None,
                 self_attn_k_injected=None,
                 self_attn_v_injected=None,
                 self_attn_cnt_k_injected=None,
                 self_attn_sty_q_injected=None,
                 self_attn_cnt_v_injected=None,
                 injection_config=None):
        x_ = self.attn1(self.norm1(x),
                       q_injected=self_attn_q_injected,
                       k_injected=self_attn_k_injected,
                       v_injected=self_attn_v_injected,
                       cnt_k_injected=self_attn_cnt_k_injected,
                       sty_q_injected=self_attn_sty_q_injected,
                       cnt_v_injected=self_attn_cnt_v_injected,
                       injection_config=injection_config,)
        x = x_ + x
        x = self.attn2(self.norm2(x), context=context) + x
        x = self.ff(self.norm3(x)) + x
        return x


class SpatialTransformer(nn.Module):
    """
    Transformer block for image-like data.
    First, project the input (aka embedding)
    and reshape to b, t, d.
    Then apply standard transformer action.
    Finally, reshape to image
    """
    def __init__(self, in_channels, n_heads, d_head,
                 depth=1, dropout=0., context_dim=None):
        super().__init__()
        self.in_channels = in_channels
        inner_dim = n_heads * d_head
        self.norm = Normalize(in_channels)

        self.proj_in = nn.Conv2d(in_channels,
                                 inner_dim,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)

        self.transformer_blocks = nn.ModuleList(
            [BasicTransformerBlock(inner_dim, n_heads, d_head, dropout=dropout, context_dim=context_dim)
                for d in range(depth)]
        )

        self.proj_out = zero_module(nn.Conv2d(inner_dim,
                                              in_channels,
                                              kernel_size=1,
                                              stride=1,
                                              padding=0))

    def forward(self,
                x,
                context=None,
                self_attn_q_injected=None,
                self_attn_k_injected=None,
                self_attn_v_injected=None,
                ## 마스크용
                self_attn_cnt_k_injected=None, 
                self_attn_sty_q_injected=None,
                self_attn_cnt_v_injected=None,
                injection_config=None):
        # note: if no context is given, cross-attention defaults to self-attention
        b, c, h, w = x.shape
        x_in = x
        x = self.norm(x)
        x = self.proj_in(x)
        x = rearrange(x, 'b c h w -> b (h w) c')

        for block in self.transformer_blocks:
            x = block(x,
                      context=context,
                      self_attn_q_injected=self_attn_q_injected,
                      self_attn_k_injected=self_attn_k_injected,
                      self_attn_v_injected=self_attn_v_injected,
                      ##마스크용
                      self_attn_cnt_k_injected=self_attn_cnt_k_injected,
                      self_attn_sty_q_injected=self_attn_sty_q_injected,
                      self_attn_cnt_v_injected=self_attn_cnt_v_injected,
                      
                      injection_config=injection_config)

            
        x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)
        x = self.proj_out(x)
        return x + x_in
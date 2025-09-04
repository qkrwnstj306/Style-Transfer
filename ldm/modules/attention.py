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


class CrossAttention(nn.Module):
    def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
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

        # 디버깅용 속성은 제거하거나 주의깊게 관리
        # self.attn = None  # 제거
        # self.q = None     # 제거
        # self.k = None     # 제거
        # self.v = None     # 제거
        # self.qk_sim = None # 제거
        
        ## generated pkl
        self.gen_pkl = False

        ## 마스크 적용
        self.sty_name = None
        self.cnt_name = None
        
        ## q, k, sim 저장용
        self.target_t_list = None
        self.layer_id = None
    
    def clear_cache(self):
        """명시적으로 캐시를 클리어하는 메서드"""
        # 속성들이 존재할 경우 제거
        attrs_to_clear = ['attn', 'q', 'k', 'v', 'qk_sim']
        for attr in attrs_to_clear:
            if hasattr(self, attr):
                delattr(self, attr)
    
    def get_batch_sim(self, q, k, num_heads, **kwargs):
        with torch.no_grad():  # gradient 계산 방지
            q = rearrange(q, "(b h) n d -> h (b n) d", h=num_heads)
            k = rearrange(k, "(b h) n d -> h (b n) d", h=num_heads)
            
            sim = torch.einsum("h i d, h j d -> h i j", q, k) * self.scale
            return sim 
    
    
    def get_batch_sim_with_mask(self, cc_sim, delta_q, delta_k, q, k, num_heads, 
                                sty_name, cnt_name, mask_path=None, attn_matrix_scale=1.0, 
                                ch=None, injection_config=None, target_t_list=None):
        
        with torch.no_grad():  # gradient 계산 방지
            q = rearrange(q, "(b h) n d -> h (b n) d", h=num_heads)
            k = rearrange(k, "(b h) n d -> h (b n) d", h=num_heads)

            sim = torch.einsum("h i d, h j d -> h i j", q, k)
            sim *= attn_matrix_scale
            sim *= self.scale

            head_num = sim.shape[0]
            pixel_size = sim.shape[1]
            h = w = int(sqrt(pixel_size))

            # 메모리 효율적인 처리
            sim_reshaped = sim.reshape(head_num, h, w, pixel_size)
            cc_sim_reshaped = cc_sim.reshape(head_num, h, w, pixel_size)
            
            delta_q = rearrange(delta_q, "(b h) n d -> h (b n) d", h=num_heads)
            delta_k = rearrange(delta_k, "(b h) n d -> h (b n) d", h=num_heads)
            max_sim = torch.einsum("h i d, h j d -> h i j", delta_q, delta_k)
            max_sim_reshaped = max_sim.reshape(head_num, h, w, pixel_size)
            
            # 중간 변수 즉시 삭제
            del delta_q, delta_k, max_sim
            
            min_cc_sim_reshaped, _ = torch.min(cc_sim_reshaped, dim=3, keepdim=True)
            max_sim_reshaped, _ = torch.max(max_sim_reshaped, dim=3, keepdim=True)
            
            length = w
            
            # mask 처리
            mask = torch.tensor(np.load(mask_path), dtype=torch.float32).cuda()
            mask[mask < 0.5] = -1.0
            mask[mask > 0.5] = 1.0
            mask = mask * ch
            
            mask = mask.unsqueeze(0).unsqueeze(0)
            mask = F.interpolate(mask, size=(h, w), mode='bilinear', align_corners=False)
            mask = mask.reshape(1, h, w, 1).to(sim.device)
            
            gradual_vanished_array = mask
            delta = min_cc_sim_reshaped - max_sim_reshaped
            gradual_vanished_mask = delta * gradual_vanished_array
            
            # in-place 연산으로 메모리 절약
            sim_reshaped[:, :length, :, :] += gradual_vanished_mask
            
            # 중간 변수 삭제
            del min_cc_sim_reshaped, max_sim_reshaped, delta, gradual_vanished_mask, mask, gradual_vanished_array
            
            sim = sim_reshaped.reshape(head_num, pixel_size, pixel_size)
            del sim_reshaped
            
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
                sty2_q_injected=None,
                sty2_k_injected=None,
                sty2_v_injected=None,
                injection_config=None,):
        
        batch, seq_len, _ = x.shape
        h = self.heads
        b = x.shape[0]
        
        # 매 forward 시작 시 캐시 클리어
        self.clear_cache()
        
        with torch.cuda.amp.autocast(enabled=False):  # FP32로 실행하여 메모리 절약
            attn_matrix_scale = 1.0
            q_mix = 0.
            is_cross = context is not None
            
            if injection_config is not None:
                attn_matrix_scale = injection_config.get('T', 1.0)
                q_mix = injection_config.get('gamma', 0.0)

            # Q 계산
            if q_injected is None:
                q = self.to_q(x)
                q = rearrange(q, 'b n (h d) -> (b h) n d', h=h)
            else:
                q_uncond = q_injected
                q_in = torch.cat([q_uncond]*b)
                q_ = self.to_q(x)
                q_ = rearrange(q_, 'b n (h d) -> (b h) n d', h=h)
                q = q_in * q_mix + q_ * (1. - 0.5)
                del q_  # 즉시 삭제
                
            context = default(context, x)

            # K 계산
            if k_injected is None:
                k = self.to_k(context)
                k = rearrange(k, 'b m (h d) -> (b h) m d', h=h)
            else:
                k_uncond = k_injected
                k = torch.cat([k_uncond]*b, dim=0)

            # V 계산
            if v_injected is None:
                v = self.to_v(context)
                v = rearrange(v, 'b m (h d) -> (b h) m d', h=h)
            else:
                v_uncond = v_injected
                v = torch.cat([v_uncond]*b, dim=0)

            # 마스크 적용
            if not self.gen_pkl:
                base_name, _ = os.path.splitext(self.cnt_name) if self.cnt_name else (None, None)
                mask_path = f"{base_name}_mask.npy" if base_name else None
                
                use_mask = (
                    self.cnt_name is not None
                    and not is_cross
                    and mask_path and os.path.exists(mask_path)
                )

                if use_mask:
                    if q_injected is not None and k_injected is not None:
                        # 중간 변수들을 효율적으로 관리
                        q_cnt = q_in if 'q_in' in locals() else q
                        k_cnt = torch.cat([cnt_k_injected]*b, dim=0)
                        v_cnt = torch.cat([cnt_v_injected]*b, dim=0)
                        k_sty_2 = torch.cat([sty2_k_injected]*b, dim=0)
                        v_sty_2 = torch.cat([sty2_v_injected]*b, dim=0)
                        
                        # cc_sim 계산
                        cc_sim = self.get_batch_sim(q=q_cnt, k=k_cnt, num_heads=h)
                        
                        # sim_1 계산
                        sim_1 = self.get_batch_sim_with_mask(
                            cc_sim=cc_sim, q=q, delta_q=q, delta_k=k, k=k,
                            num_heads=h, sty_name=self.sty_name, cnt_name=self.cnt_name,
                            mask_path=mask_path, attn_matrix_scale=attn_matrix_scale,
                            ch=-1.0, injection_config=injection_config,
                            target_t_list=self.target_t_list,
                        )
                        
                        # sim_2 계산
                        sim_2 = self.get_batch_sim_with_mask(
                            cc_sim=cc_sim, q=q, delta_q=q, delta_k=k_sty_2, k=k_sty_2,
                            num_heads=h, sty_name=self.sty_name, cnt_name=self.cnt_name,
                            mask_path=mask_path, attn_matrix_scale=attn_matrix_scale,
                            ch=1.0,
                        )
                        
                        # Concatenation 대신 sequential 처리 고려
                        # 메모리 효율을 위해 chunk 처리
                        cat_sim = torch.cat((sim_1, sim_2, cc_sim), 2)
                        cat_v = torch.cat((v, v_sty_2, v_cnt), 1)
                        
                        # 중간 변수 즉시 삭제
                        del sim_1, sim_2, cc_sim, q_cnt, k_cnt, v_cnt, k_sty_2, v_sty_2
                        
                        cat_sim = cat_sim.softmax(-1)
                        cat_out = einsum('b i j, b j d -> b i d', cat_sim, cat_v)
                        
                        del cat_sim, cat_v  # 즉시 삭제
                        
                        out = rearrange(cat_out, 'h (b n) d -> b n (h d)', h=h, b=b)
                        del cat_out
                        
                    else:
                        # style injection이 없는 경우
                        sim = einsum('b i d, b j d -> b i j', q, k)
                        sim *= attn_matrix_scale * self.scale
                        attn = sim.softmax(dim=-1)
                        out = einsum('b i j, b j d -> b i d', attn, v)
                        out = rearrange(out, '(b h) n d -> b n (h d)', h=h)
                        del sim, attn  # 즉시 삭제
                        
                else:
                    # 마스크 미적용
                    sim = einsum('b i d, b j d -> b i j', q, k)
                    if q_injected is not None or k_injected is not None:
                        sim *= attn_matrix_scale    
                    sim *= self.scale
                    attn = sim.softmax(dim=-1)
                    out = einsum('b i j, b j d -> b i d', attn, v)
                    out = rearrange(out, '(b h) n d -> b n (h d)', h=h)
                    del sim, attn  # 즉시 삭제
                    
            else:
                # 원본 처리
                sim = einsum('b i d, b j d -> b i j', q, k)
                sim *= self.scale
                
                if exists(mask):
                    mask = rearrange(mask, 'b ... -> b (...)')
                    max_neg_value = -torch.finfo(sim.dtype).max
                    mask = repeat(mask, 'b j -> (b h) () j', h=h)
                    sim.masked_fill_(~mask, max_neg_value)
                
                attn = sim.softmax(dim=-1)
                out = einsum('b i j, b j d -> b i d', attn, v)
                out = rearrange(out, '(b h) n d -> b n (h d)', h=h)
                del sim, attn  # 즉시 삭제
            
            # q, k, v 삭제
            del q, k, v
            
            return self.to_out(out)


class BasicTransformerBlock(nn.Module):
    def __init__(self, dim, n_heads, d_head, dropout=0., context_dim=None, gated_ff=True, checkpoint=True):
        super().__init__()
        self.attn1 = CrossAttention(query_dim=dim, heads=n_heads, dim_head=d_head, dropout=dropout)
        self.ff = FeedForward(dim, dropout=dropout, glu=gated_ff)
        self.attn2 = CrossAttention(query_dim=dim, context_dim=context_dim,
                                    heads=n_heads, dim_head=d_head, dropout=dropout)
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
                self_attn_sty2_q_injected=None,
                self_attn_sty2_k_injected=None,
                self_attn_sty2_v_injected=None,
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
                                          self_attn_sty2_q_injected,
                                          self_attn_sty2_k_injected,
                                          self_attn_sty2_v_injected,
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
                 self_attn_sty2_q_injected=None,
                 self_attn_sty2_k_injected=None,
                 self_attn_sty2_v_injected=None,
                 injection_config=None):
        
        x_ = self.attn1(self.norm1(x),
                       q_injected=self_attn_q_injected,
                       k_injected=self_attn_k_injected,
                       v_injected=self_attn_v_injected,
                       cnt_k_injected=self_attn_cnt_k_injected,
                       sty_q_injected=self_attn_sty_q_injected,
                       cnt_v_injected=self_attn_cnt_v_injected,
                       sty2_q_injected=self_attn_sty2_q_injected,
                       sty2_k_injected=self_attn_sty2_k_injected,
                       sty2_v_injected=self_attn_sty2_v_injected,
                       injection_config=injection_config,)
        x = x_ + x
        del x_  # 중간 변수 즉시 삭제
        
        x_ = self.attn2(self.norm2(x), context=context)
        x = x_ + x
        del x_
        
        x_ = self.ff(self.norm3(x))
        x = x_ + x
        del x_
        
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
                self_attn_cnt_k_injected=None, 
                self_attn_sty_q_injected=None,
                self_attn_cnt_v_injected=None,
                self_attn_sty2_q_injected=None,
                self_attn_sty2_k_injected=None,
                self_attn_sty2_v_injected=None,
                injection_config=None):
        
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
                      self_attn_cnt_k_injected=self_attn_cnt_k_injected,
                      self_attn_sty_q_injected=self_attn_sty_q_injected,
                      self_attn_cnt_v_injected=self_attn_cnt_v_injected,
                      self_attn_sty2_q_injected=self_attn_sty2_q_injected,
                      self_attn_sty2_k_injected=self_attn_sty2_k_injected,
                      self_attn_sty2_v_injected=self_attn_sty2_v_injected,
                      injection_config=injection_config)

        x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)
        x = self.proj_out(x)
        return x + x_in
import torch
import types

def high_freq_filter(h, radius_ratio=0.2):
    #tensor: (B, C, H, W) 형태의 feature map (복소수 변환 후 실수부 복원)
    #radius_ratio : 저주파 반경 비율, 이 반경 이내는 0(제거)으로 처리

    orig_dtype = h.dtype
    h = h.to(torch.float32)

    B,C,H,W = h.shape
    # 2D FFT : (B,C,H,W) > (B,C,H,W), 복소수
    fft = torch.fft.fft2(h, norm='ortho') #입력 feature map에 대해 2D FFT 수행, 공간 도메인의 신호 > 주파수 도메인
    fft_shift = torch.fft.fftshift(fft) #주파수 중심 이동 (저주파 성분이 중앙에 오도록)

    #중심점 좌표
    cy, cx = H//2, W//2
    radius = int(min(H,W) * radius_ratio) #중앙에서 radius 크기 원형 영역이 저주파 영역임을 의미

    #저주파 마스크 생성 (중심에 radius 크기 원형 영역만 0, 나머지는 1) (마스크 크기 H,W)
    y = torch.arange(H, device=h.device).view(-1,1)
    x = torch.arange(W, device=h.device).view(1,-1)
    dist = (y-cy)**2+(x-cx)**2
    mask = torch.ones((H,W), device = h.device)
    mask[dist<radius**2] = 0
    mask = mask.unsqueeze(0).unsqueeze(0)

    #고주파만 남기기
    fft_filtered = fft_shift * mask

    #다시 원래 위치로 이동 후 inverse FFT
    fft_ifftshift = torch.fft.ifftshift(fft_filtered) #다시 주파수 배치로 되돌림
    filtered = torch.fft.ifft2(fft_ifftshift, norm='ortho') #주파수 도메인 > 공간 도메인 (고주파 성분만 남은 feature map(복소수))

    #실수부 반환(고주파 성분)
    return filtered.real.to(orig_dtype)

def make_content_injection_schedule(ddim_timesteps, alpha=0.4):
    """
    앞쪽 timestep 중 일정 비율(alpha)만 injection 타겟으로 지정
    """
    T = len(ddim_timesteps)
    return ddim_timesteps[:int(alpha * T)]


def patch_decoder_resblocks_h_and_cnt_hf(unet, schedule, residuals_all):
    """
    ResBlock의 _forward를 덮어써서 고주파 주입
    mode: ['h_only', 'skip_only', 'both', 'none']
    where: ['none', 'add']
    """

    for block_id in range(6, 12):
        if block_id >= len(unet.output_blocks):
            break

        for module in reversed(unet.output_blocks[block_id]):
            if module.__class__.__name__.endswith("ResBlock"):
                module.block_id = block_id
                orig_forward = module._forward

                def wrapped_forward(self, x, emb, out_layers_injected=None, _orig=orig_forward):
                    out_stylized = _orig(x, emb, out_layers_injected)
                    t = getattr(self, "ri_timestep", None)
                    key_h = f"output_block_{self.block_id}_cnt_h"
                    #key_skip = f"output_block_{self.block_id}_cnt_skip"
                    #key_all = f"output_block_{self.block_id}_residual"
                    out_res = out_stylized

                    if t in schedule and t in residuals_all:
                        h_cnt = residuals_all[t].get(key_h, None)
                        #skip_cnt = residuals_all[t].get(key_skip, None)

                        #if h_cnt is not None and skip_cnt is not None:
                        if h_cnt is not None:
                            h_cnt = h_cnt.to(out_stylized.device)
                            print(f"[DEBUG] h_cnt type at t={t}, key={key_h}:", type(h_cnt))

                            #skip_cnt = skip_cnt.to(out_stylized.device)

                            # 고주파 필터 적용
                            h_cnt_hf = high_freq_filter(h_cnt)
                            #skip_cnt_hf = high_freq_filter(skip_cnt)

                            skip_stylized = self.out_skip
                            h_stylized = self.out_h

                            out_res = skip_stylized + h_stylized + h_cnt_hf
                    return out_res

                module._forward = types.MethodType(wrapped_forward, module)
                break
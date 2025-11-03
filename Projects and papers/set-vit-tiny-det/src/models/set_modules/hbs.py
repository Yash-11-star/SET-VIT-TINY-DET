import torch
import torch.nn as nn
import torch.nn.functional as F

def odd_kernel_from_stride(stride: int) -> int:
    # SET paper uses log2(stride) mapping to odd K; simple version:
    # K = max(3, 2 * floor(log2(S)/2) + 1)
    import math
    k = max(3, int(2 * (math.floor(math.log2(max(1, stride)) / 2)) + 1))
    if k % 2 == 0: k += 1
    return k

class HBS(nn.Module):
    """
    Hierarchical Background Smoothing for transformer feature maps.
    Operates per-scale on P (B,C,H,W) given a binary mask M (B,1,H,W).
    Only smooths background (1-M), preserves foreground.
    """
    def __init__(self, channels: int, reduction: int = 4, kernel_size: int = 3):
        super().__init__()
        hidden = max(1, channels // reduction)
        padding = kernel_size // 2
        self.reduce = nn.Conv2d(channels, hidden, kernel_size, padding=padding, bias=True)
        self.expand = nn.Conv2d(hidden, channels, kernel_size, padding=padding, bias=True)
        self.act = nn.ReLU(inplace=True)

    def forward(self, P, M):
        # M: 1 on foreground; we smooth background
        P_fg = P * M
        P_bg = P * (1 - M)
        smooth = self.expand(self.act(self.reduce(P_bg))) + P_bg
        return P_fg + smooth

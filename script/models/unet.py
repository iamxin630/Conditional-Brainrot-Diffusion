"""
Conditional UNet for DDPM/DDIM image generation.

Architecture:
  - Sinusoidal timestep embedding → MLP
  - Condition embedding: animal(11) + object(11) + pair(101) → cond vector
  - Encoder (down) → Middle → Decoder (up) with skip connections
  - ResBlocks use Adaptive Group Norm (scale-shift) conditioned on time + cond
  - Self-attention at specified resolutions (16×16 and 8×8 by default)

Target parameter count: ~62M (matching HW6 baseline).
"""

import math
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class SinusoidalPosEmb(nn.Module):
    """Sinusoidal positional embedding for timesteps."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        device = t.device
        half = self.dim // 2 # # 計算衰減係數的基準分母對數
        emb = math.log(10000) / (half - 1) # # 設定一個分母，用來控制頻率衰減得快不快
        emb = torch.exp(torch.arange(half, device=device) * -emb) # # 計算每個位置的頻率
        emb = t[:, None].float() * emb[None, :] # # 把時間和頻率相乘，得到每個位置的相位值
        emb = torch.cat([emb.sin(), emb.cos()], dim=-1) # # 得到最終的 time embedding
        return emb

# 把特徵圖（圖片）的長和寬縮小一半，好讓模型能夠看得更廣（捕捉全局大範圍的語意），同時還能節約記憶體。
class Downsample(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)

# 把特徵圖的長和寬放大為原來的 2 倍，逐步恢復圖片的細節解析度。
class Upsample(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        return self.conv(x)


# ---------------------------------------------------------------------------
# ResBlock with Adaptive Group Norm (scale-shift)
# ---------------------------------------------------------------------------

class ResBlock(nn.Module):
    """
    ResNet block with AdaGN.

    If use_scale_shift_norm:
        out = GroupNorm(h) * (1 + scale) + shift
    Otherwise:
        out = GroupNorm(h) + emb_proj
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        emb_dim: int,
        num_groups: int = 32,
        dropout: float = 0.0,
        use_scale_shift_norm: bool = True,
    ):
        super().__init__()
        self.use_scale_shift_norm = use_scale_shift_norm
        
        # 準備兩層普通的卷積，負責處理「圖片本身」的特徵
        self.norm1 = nn.GroupNorm(num_groups, in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)

        self.norm2 = nn.GroupNorm(num_groups, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)

        # Embedding projection: if scale-shift, project to 2*out_channels
        proj_out = out_channels * 2 if use_scale_shift_norm else out_channels
        self.emb_proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(emb_dim, proj_out),
        )

        self.dropout = nn.Dropout(dropout)

        # Skip connection when channel count changes
        # 如果輸入通道數不等於輸出通道數，就用 1x1 卷積進行通道變換；否則直接原樣通過 (Identity)
        if in_channels != out_channels:
            self.skip_conv = nn.Conv2d(in_channels, out_channels, 1)
        else:
            self.skip_conv = nn.Identity()

    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        # 基礎特徵提取 (第 1 層卷積)
        h = self.norm1(x) # 標準化
        h = F.silu(h)     # 激活
        h = self.conv1(h) # 卷積，提取初步特徵

        # Inject time + condition embedding
        # 補上兩個空的維度，變成四維，讓時間訊息可以「疊」在圖片特徵上
        emb_out = self.emb_proj(emb)[:, :, None, None] 

        # scale and shift
        # 對半切開，分成了兩個獨立的四維向量：scale 和 shift。
        if self.use_scale_shift_norm: 
            scale, shift = emb_out.chunk(2, dim=1) # 分成兩份
            h = self.norm2(h) * (1 + scale) + shift # 全部乘以 (1 + scale 的第 1 個數字)，然後全部加上 shift 的第 1 個數字
        else:
            h = self.norm2(h + emb_out) # 單純把條件 emb_out 加進影像特徵後做標準化。

        # 深度特徵提取 (第 2 層卷積)
        h = F.silu(h)
        h = self.dropout(h)  # 隨機關掉一些神經元，防止模型死背
        h = self.conv2(h)  # 卷積，提取深層特徵

        return h + self.skip_conv(x)  # 最後加上跳躍連接 (Shortcut Connection)


# ---------------------------------------------------------------------------
# Self-Attention
# ---------------------------------------------------------------------------

class SelfAttention(nn.Module):
    """Multi-head self-attention with GroupNorm pre-norm."""

    def __init__(self, channels: int, num_heads: int = 4, num_groups: int = 32):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        assert channels % num_heads == 0

        self.norm = nn.GroupNorm(num_groups, channels)
        self.qkv = nn.Conv1d(channels, channels * 3, 1)
        self.out_proj = nn.Conv1d(channels, channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        h = self.norm(x).view(B, C, H * W)  # [B, C, N]

        qkv = self.qkv(h)  # [B, 3C, N]
        q, k, v = qkv.chunk(3, dim=1)  # each [B, C, N]

        # Reshape for multi-head attention
        q = q.view(B, self.num_heads, self.head_dim, H * W).permute(0, 1, 3, 2)  # [B, heads, N, d]
        k = k.view(B, self.num_heads, self.head_dim, H * W).permute(0, 1, 3, 2)
        v = v.view(B, self.num_heads, self.head_dim, H * W).permute(0, 1, 3, 2)

        # Efficient attention (FlashAttention when available)
        out = F.scaled_dot_product_attention(q, k, v)  # [B, heads, N, d]

        out = out.permute(0, 1, 3, 2).contiguous().view(B, C, H * W)  # [B, C, N]
        out = self.out_proj(out)
        return out.view(B, C, H, W) + x


# ---------------------------------------------------------------------------
# Combined ResBlock + optional Attention as a single unit
# ---------------------------------------------------------------------------

class ResAttnBlock(nn.Module):
    """A ResBlock optionally followed by SelfAttention. Treated as one unit for skip tracking."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        emb_dim: int,
        num_groups: int = 32,
        dropout: float = 0.0,
        use_scale_shift_norm: bool = True,
        use_attention: bool = False,
    ):
        super().__init__()
        self.resblock = ResBlock(in_channels, out_channels, emb_dim, num_groups, dropout, use_scale_shift_norm)
        self.attn = SelfAttention(out_channels, num_heads=max(1, out_channels // 64), num_groups=num_groups) if use_attention else None

    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        h = self.resblock(x, emb)
        if self.attn is not None:
            h = self.attn(h)
        return h


# ---------------------------------------------------------------------------
# Conditional UNet
# ---------------------------------------------------------------------------

class ConditionalUNet(nn.Module):
    """
    Conditional UNet for diffusion model.

    Uses a clean skip-connection pattern:
      - Encoder: each ResAttnBlock output is appended to skip list
      - Downsample outputs are also appended
      - Decoder: each ResAttnBlock pops a skip, concats, and processes
      - Upsample after each level

    Parameters
    ----------
    image_size : int
        Input spatial resolution (e.g. 64).
    in_channels : int
        Input image channels (3 for RGB).
    base_channels : int
        Base channel count (128 for ~62M params).
    channel_mults : list of int
        Channel multipliers per resolution level.
    num_res_blocks : int
        Number of ResBlocks per level in the encoder.
    attention_resolutions : list of int
        Resolutions at which to apply self-attention.
    dropout : float
        Dropout rate in ResBlocks.
    time_dim : int
        Dimension of the timestep embedding.
    cond_dim : int
        Dimension of the condition embedding.
    num_animals, num_objects, num_pairs : int
        Number of classes (excluding null).
    animal_emb_dim, object_emb_dim, pair_emb_dim : int
        Embedding dimensions.
    num_groups : int
        Groups for GroupNorm.
    use_scale_shift_norm : bool
        Use adaptive group norm in ResBlocks.
    """

    def __init__(
        self,
        image_size: int = 64,
        in_channels: int = 3,
        base_channels: int = 128,
        channel_mults: List[int] = [1, 2, 2, 4],
        num_res_blocks: int = 2,
        attention_resolutions: List[int] = [16, 8],
        dropout: float = 0.1,
        time_dim: int = 512,
        cond_dim: int = 512,
        num_animals: int = 10,
        num_objects: int = 10,
        num_pairs: int = 100,
        animal_emb_dim: int = 256,
        object_emb_dim: int = 256,
        pair_emb_dim: int = 512,
        num_groups: int = 32,
        use_scale_shift_norm: bool = True,
    ):
        super().__init__()
        self.image_size = image_size
        emb_dim = time_dim
        assert time_dim == cond_dim, "time_dim and cond_dim must match for addition"

        # --- Time embedding ---
        self.time_embed = nn.Sequential(
            SinusoidalPosEmb(base_channels),
            nn.Linear(base_channels, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )

        # --- Condition embedding ---
        self.animal_emb = nn.Embedding(num_animals + 1, animal_emb_dim)
        self.object_emb = nn.Embedding(num_objects + 1, object_emb_dim)
        self.pair_emb = nn.Embedding(num_pairs + 1, pair_emb_dim)
        self.cond_proj = nn.Sequential(
            nn.Linear(animal_emb_dim + object_emb_dim, cond_dim),
            nn.SiLU(),
            nn.Linear(cond_dim, cond_dim),
        )

        # --- Input conv ---
        self.input_conv = nn.Conv2d(in_channels, base_channels, 3, padding=1)

        # --- Build encoder (down path) ---
        self.down_blocks = nn.ModuleList()
        self.down_samples = nn.ModuleList()

        # Track channel sizes at each skip position for the decoder
        skip_channels = [base_channels]  # after input_conv
        ch = base_channels
        cur_res = image_size

        for level, mult in enumerate(channel_mults):
            out_ch = base_channels * mult
            for _ in range(num_res_blocks):
                use_attn = (cur_res in attention_resolutions)
                self.down_blocks.append(
                    ResAttnBlock(ch, out_ch, emb_dim, num_groups, dropout, use_scale_shift_norm, use_attn)
                )
                ch = out_ch
                skip_channels.append(ch)

            if level < len(channel_mults) - 1:
                self.down_samples.append(Downsample(ch))
                skip_channels.append(ch)
                cur_res //= 2

        # --- Middle block ---
        self.mid_block1 = ResBlock(ch, ch, emb_dim, num_groups, dropout, use_scale_shift_norm)
        self.mid_attn = SelfAttention(ch, num_heads=max(1, ch // 64), num_groups=num_groups)
        self.mid_block2 = ResBlock(ch, ch, emb_dim, num_groups, dropout, use_scale_shift_norm)

        # --- Build decoder (up path) ---
        self.up_blocks = nn.ModuleList()
        self.up_samples = nn.ModuleList()

        for level in reversed(range(len(channel_mults))):
            out_ch = base_channels * channel_mults[level]
            level_res = image_size // (2 ** level)

            for i in range(num_res_blocks + 1):  # +1 to process the extra skip
                skip_ch = skip_channels.pop()
                use_attn = (level_res in attention_resolutions)
                self.up_blocks.append(
                    ResAttnBlock(ch + skip_ch, out_ch, emb_dim, num_groups, dropout, use_scale_shift_norm, use_attn)
                )
                ch = out_ch

            if level > 0:
                self.up_samples.append(Upsample(ch))

        # --- Output ---
        self.out_norm = nn.GroupNorm(num_groups, ch)
        self.out_conv = nn.Conv2d(ch, in_channels, 3, padding=1)
        nn.init.zeros_(self.out_conv.weight)
        nn.init.zeros_(self.out_conv.bias)

        # Store for reference
        self._num_down_blocks_per_level = num_res_blocks
        self._num_levels = len(channel_mults)

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        animal_ids: torch.Tensor,
        object_ids: torch.Tensor,
        pair_ids: torch.Tensor,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        x : [B, C, H, W]  noisy image
        t : [B]  timestep indices (0 to T-1)
        animal_ids : [B]  animal class ids (0-9, or 10 for null)
        object_ids : [B]  object class ids (0-9, or 10 for null)
        pair_ids : [B]  pair ids (0-99, or 100 for null)

        Returns
        -------
        [B, C, H, W]  predicted noise
        """
        # Compute embeddings
        time_emb = self.time_embed(t)
        animal_e = self.animal_emb(animal_ids)
        object_e = self.object_emb(object_ids)
        pair_e = self.pair_emb(pair_ids)
        cond_emb = self.cond_proj(torch.cat([animal_e, object_e], dim=-1)) + pair_e
        emb = time_emb + cond_emb

        # --- Encoder ---
        h = self.input_conv(x)
        skips = [h]

        block_idx = 0
        ds_idx = 0
        for level in range(self._num_levels):
            for _ in range(self._num_down_blocks_per_level):
                h = self.down_blocks[block_idx](h, emb)
                skips.append(h)
                block_idx += 1
            if level < self._num_levels - 1:
                h = self.down_samples[ds_idx](h)
                skips.append(h)
                ds_idx += 1

        # --- Middle ---
        h = self.mid_block1(h, emb)
        h = self.mid_attn(h)
        h = self.mid_block2(h, emb)

        # --- Decoder ---
        block_idx = 0
        us_idx = 0
        for level in reversed(range(self._num_levels)):
            for _ in range(self._num_down_blocks_per_level + 1):
                skip = skips.pop()
                h = torch.cat([h, skip], dim=1)
                h = self.up_blocks[block_idx](h, emb)
                block_idx += 1
            if level > 0:
                h = self.up_samples[us_idx](h)
                us_idx += 1

        # --- Output ---
        h = self.out_norm(h)
        h = F.silu(h)
        h = self.out_conv(h)
        return h


def build_unet_from_config(config: dict) -> ConditionalUNet:
    """Construct a ConditionalUNet from a JSON config dict."""
    unet_cfg = config["unet"]
    cond_cfg = config["condition"]
    return ConditionalUNet(
        image_size=unet_cfg["image_size"],
        in_channels=unet_cfg["in_channels"],
        base_channels=unet_cfg["base_channels"],
        channel_mults=unet_cfg["channel_mults"],
        num_res_blocks=unet_cfg["num_res_blocks"],
        attention_resolutions=unet_cfg["attention_resolutions"],
        dropout=unet_cfg["dropout"],
        time_dim=unet_cfg["time_dim"],
        cond_dim=unet_cfg["cond_dim"],
        num_animals=cond_cfg["num_animals"],
        num_objects=cond_cfg["num_objects"],
        num_pairs=cond_cfg["num_pairs"],
        animal_emb_dim=cond_cfg["animal_emb_dim"],
        object_emb_dim=cond_cfg["object_emb_dim"],
        pair_emb_dim=cond_cfg["pair_emb_dim"],
        num_groups=unet_cfg["num_groups"],
        use_scale_shift_norm=unet_cfg["use_scale_shift_norm"],
    )

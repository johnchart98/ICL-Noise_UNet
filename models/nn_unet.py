"""
A compact, single-file PyTorch implementation of an nnU-Net-like architecture.
Features:
- Configurable for 2D or 3D (dim=2 or dim=3)
- Encoder-decoder (U-Net) with InstanceNorm, LeakyReLU
- Optional dropout
- Upsampling with ConvTranspose (learned) or nearest+conv
- Deep supervision heads (multi-scale outputs)
- Example usage at the bottom

Notes:
- This is an educational, practical reimplementation inspired by nnU-Net design choices,
  not a line-by-line clone of the original project.
- For production, combine with nnU-Net training schedules/data-augmentation and careful
  hyperparameter tuning.
"""

from typing import Tuple, List, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


def get_conv(ndim: int):
    return {2: nn.Conv2d, 3: nn.Conv3d}[ndim]


def get_conv_transpose(ndim: int):
    return {2: nn.ConvTranspose2d, 3: nn.ConvTranspose3d}[ndim]


def get_norm(ndim: int):
    return {2: nn.InstanceNorm2d, 3: nn.InstanceNorm3d}[ndim]


class ConvBlock(nn.Module):
    """Two-conv block used in nnU-Net style networks."""

    def __init__(self, in_ch, out_ch, ndim=2, kernel_size=3, padding=1, dropout=0.0):
        super().__init__()
        Conv = get_conv(ndim)
        Norm = get_norm(ndim)
        self.conv1 = Conv(in_ch, out_ch, kernel_size=kernel_size, padding=padding, bias=True)
        self.norm1 = Norm(out_ch, affine=False)
        self.act1 = nn.LeakyReLU(negative_slope=0.01, inplace=True)

        self.conv2 = Conv(out_ch, out_ch, kernel_size=kernel_size, padding=padding, bias=True)
        self.norm2 = Norm(out_ch, affine=False)
        self.act2 = nn.LeakyReLU(negative_slope=0.01, inplace=True)

        self.dropout = nn.Dropout3d(p=dropout) if ndim == 3 and dropout > 0 else (
            nn.Dropout2d(p=dropout) if ndim == 2 and dropout > 0 else nn.Identity()
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.act1(x)
        x = self.dropout(x)
        x = self.conv2(x)
        x = self.norm2(x)
        x = self.act2(x)
        return x


class Downsample(nn.Module):
    """Downsampling by strided convolution (as in many nnU-Net variants)"""

    def __init__(self, in_ch, out_ch, ndim=2):
        super().__init__()
        Conv = get_conv(ndim)
        self.down = Conv(in_ch, out_ch, kernel_size=2, stride=2)

    def forward(self, x):
        return self.down(x)


class Upsample(nn.Module):
    """Upsampling - transposed conv by default; fallback to interpolate+conv if requested."""

    def __init__(self, in_ch, out_ch, ndim=2, mode='transpose'):
        super().__init__()
        if mode == 'transpose':
            ConvT = get_conv_transpose(ndim)
            self.up = ConvT(in_ch, out_ch, kernel_size=2, stride=2)
        else:
            # interpolate then 1x1 conv
            Conv = get_conv(ndim)
            self.up = nn.Sequential(nn.Upsample(scale_factor=2, mode='nearest'), Conv(in_ch, out_ch, kernel_size=1))

    def forward(self, x):
        return self.up(x)


class NnUNet(nn.Module):
    """
    Configurable nnU-Net-like architecture.

    Args:
        in_channels: number of input channels
        out_channels: number of segmentation classes (logits)
        base_num_features: number of filters in the first stage (commonly 32)
        num_pool: number of downsampling operations (depth)
        ndim: 2 or 3
        deep_supervision: whether to produce multi-scale outputs
        dropout: dropout probability in conv blocks
        up_mode: 'transpose' or 'interpolate'
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        base_num_features: int = 32,
        num_pool: int = 5,
        ndim: int = 2,
        deep_supervision: bool = True,
        dropout: float = 0.0,
        up_mode: str = 'transpose',
    ):
        super().__init__()
        assert ndim in (2, 3), "ndim must be 2 or 3"
        self.ndim = ndim
        self.deep_supervision = deep_supervision
        Conv = get_conv(ndim)

        # build encoder
        in_ch = in_channels
        feats = []
        num_features = base_num_features
        self.enc_blocks = nn.ModuleList()
        self.downs = nn.ModuleList()
        for d in range(num_pool):
            self.enc_blocks.append(ConvBlock(in_ch, num_features, ndim=ndim, dropout=dropout))
            feats.append(num_features)
            in_ch = num_features
            num_features = min(num_features * 2, 320)  # cap like nnU-Net
            if d != num_pool - 1:
                # downsample between stages
                self.downs.append(Downsample(in_ch, num_features, ndim=ndim))
                in_ch = num_features

        # bottleneck (last encoder block is the bottleneck here)
        self.bottleneck_idx = len(self.enc_blocks) - 1

        # build decoder
        self.ups = nn.ModuleList()
        self.dec_blocks = nn.ModuleList()
        # reversed features for decoder
        for i in range(num_pool - 1):
            # features from deepest to shallowest (e.g., [16, 32, 64, 128])
            skip_ch = feats[-(i + 2)]  # feature map from encoder skip
            bottleneck_ch = feats[-(i + 1)]  # input channels from previous level (decoder input)

            # upsample from bottleneck_ch -> skip_ch
            self.ups.append(Upsample(bottleneck_ch, skip_ch, ndim=ndim, mode=up_mode))

            # after concatenation: in_ch = skip_ch + skip_ch
            self.dec_blocks.append(ConvBlock(skip_ch * 2, skip_ch, ndim=ndim, dropout=dropout))

        # final conv(s)
        self.seg_heads = nn.ModuleList()
        out_in_ch = feats[0]
        self.final_conv = Conv(out_in_ch, out_channels, kernel_size=1)

        # deep supervision heads for intermediate decoder outputs
        if self.deep_supervision:
            # produce a head for each decoder block (except final) -> follow nnU-Net: multiple auxiliary outputs
            for i, f in enumerate(reversed(feats[:-1])):  # one head per decoder stage
                self.seg_heads.append(nn.Sequential(Conv(f, out_channels, kernel_size=1)))

    def forward(self, x, context=None,context_out=None):
        enc_features = []
        cur = x

        # ----- Encoder -----
        for i, enc in enumerate(self.enc_blocks):
            cur = enc(cur)
            enc_features.append(cur)
            if i < len(self.downs):  # not last level
                cur = self.downs[i](cur)

        # cur is the bottleneck feature map (deepest level)
        bottleneck = cur

        # ----- Decoder -----
        deep_outputs = []
        cur = bottleneck

        for i in range(len(self.ups)):
            up = self.ups[i]
            dec = self.dec_blocks[i]

            # get corresponding skip feature (mirror of encoder)
            skip = enc_features[-(i + 2)]  # -1 is bottleneck, so -(i+2) is skip above it

            # upsample
            cur = up(cur)

            # make sure sizes match (due to rounding in convs)
            if cur.shape[2:] != skip.shape[2:]:
                cur = F.interpolate(
                    cur,
                    size=skip.shape[2:],
                    mode='trilinear' if self.ndim == 3 else 'bilinear',
                    align_corners=False
                )

            # concatenate skip connection
            cur = torch.cat([skip, cur], dim=1)

            # convolutional refinement
            cur = dec(cur)

            # optionally add deep supervision output
            if self.deep_supervision and i < len(self.seg_heads):
                out = self.seg_heads[i](cur)
                deep_outputs.append(out)

        # ----- Final segmentation head -----
        seg = self.final_conv(cur)

        # ----- Deep Supervision -----
        if not self.deep_supervision:
            return seg  # single output

        # if deep supervision: upsample auxiliary outputs and average
        outs = [seg]
        final_size = seg.shape[2:]

        for d_out in deep_outputs:
            if d_out.shape[2:] != final_size:
                d_out = F.interpolate(
                    d_out,
                    size=final_size,
                    mode='trilinear' if self.ndim == 3 else 'bilinear',
                    align_corners=False
                )
            outs.append(d_out)

        # Combine all deep supervision outputs (mean)
        stacked = torch.stack(outs, dim=0)  # [num_outputs, B, C, H, W]
        seg_final = torch.mean(stacked, dim=0)

        return seg_final


# Example usage
if __name__ == '__main__':
    # 2D example
    model2d = NnUNet(in_channels=1, out_channels=3, base_num_features=16, num_pool=4, ndim=2, deep_supervision=True)
    x2 = torch.randn(2, 1, 256, 256)
    y2 = model2d(x2)
    print('2D out:', y2.shape)  # expect (2, 3, 256, 256)

    # 3D example
    model3d = NnUNet(in_channels=1, out_channels=2, base_num_features=16, num_pool=4, ndim=3, deep_supervision=True)
    x3 = torch.randn(1, 1, 64, 128, 128)
    y3 = model3d(x3)
    print('3D out:', y3.shape)  # expect (1, 2, 64, 128, 128)

    # quick parameter count
    def count_params(m):
        return sum(p.numel() for p in m.parameters() if p.requires_grad)

    print('params 2D:', count_params(model2d))
    print('params 3D:', count_params(model3d))

import math

import torch
from torch import nn


class TimeEmbedding(nn.Module):
    def __init__(self, n_channels: int):
        super().__init__()
        self.n_channels = n_channels

        self.proj = nn.Sequential(
            nn.Linear(self.n_channels // 4, self.n_channels), nn.GELU(), nn.Linear(self.n_channels, self.n_channels)
        )

    def forward(self, t: torch.Tensor):
        half_dim = self.n_channels // 8
        emb = math.log(10_000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device) * -emb)
        emb = t[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=1)

        emb = self.proj(emb)

        return emb


def same_conv(in_channels: int, out_channels: int) -> nn.Conv2d:
    return nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1)


class ResNetBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()

        self.main_branch = nn.Sequential(
            nn.GroupNorm(num_groups=4, num_channels=channels),
            nn.SiLU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups=4, num_channels=channels),
            nn.SiLU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
        )

        self.skip_branch = nn.Conv2d(channels, channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.main_branch(x) + self.skip_branch(x)


class UNetBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, maxpool: bool = False, upsample: bool = False) -> None:
        super().__init__()

        assert not (maxpool and upsample), "maxpool and upsample are mutually exclusive."

        if maxpool:
            self.out = nn.MaxPool2d(kernel_size=2)
        elif upsample:
            self.out = nn.ConvTranspose2d(out_channels, out_channels // 2, kernel_size=2, stride=2)
        else:
            self.out = nn.Identity()

        self.time_proj = nn.Linear(256, out_channels)

        self.in_conv = same_conv(in_channels, out_channels)
        self.blocks = nn.ModuleList([ResNetBlock(out_channels)])

    def forward(self, x: torch.Tensor, time_embs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        time_embs = self.time_proj(time_embs)
        x = self.in_conv(x)
        x += time_embs.reshape(x.size(0), x.size(1), 1, 1)
        for block in self.blocks:
            x = block(x)
        x_res = x
        x = self.out(x)
        return x_res, x


class SmolUNet(nn.Module):
    def __init__(self, in_channels: int, out_channels: list[int] = [64, 128, 256, 512, 1024]) -> None:
        super().__init__()

        channels = [in_channels] + out_channels[:-1]

        self.time_embedding = TimeEmbedding(256)

        self.downs = nn.ModuleList([UNetBlock(ic, oc, maxpool=True) for ic, oc in zip(channels[:-1], channels[1:])])
        self.latent_up = UNetBlock(out_channels[-2], out_channels[-1], upsample=True)
        self.ups = nn.ModuleList([UNetBlock(c * 2, c, upsample=c != out_channels[0]) for c in out_channels[::-1][1:]])
        self.out = same_conv(out_channels[0], in_channels)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        time_embs = self.time_embedding(t)
        res = []
        for down in self.downs:
            x_res, x = down(x, time_embs)
            res.append(x_res)

        _, x = self.latent_up(x, time_embs)

        for xr, up in zip(res[::-1], self.ups):
            x = torch.cat([xr, x], dim=1)
            _, x = up(x, time_embs)

        return self.out(x)


def main():
    smol_unet = SmolUNet(in_channels=3)
    x = torch.randn(8, 3, 32, 32)
    t = torch.randint(0, 200, size=(x.size(0),))
    print(smol_unet(x, t).shape)


if __name__ == "__main__":
    main()

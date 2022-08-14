import math

import torch
from torch import nn

# class SinusoidalPositionEmbeddings(nn.Module):
#     def __init__(self, dim: int):
#         super().__init__()
#         self.dim = dim

#     def forward(self, t: torch.Tensor):
#         embeddings = math.log(10000) / (self.dim // 2 - 1)
#         embeddings = torch.exp(torch.arange(self.dim // 2, device=t.device) * -embeddings)
#         # embeddings = t[:, None] * embeddings[None, :]
#         embeddings = t.unsqueeze(1) * embeddings.unsqueeze(0)
#         embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
#         return embeddings


class TimeEmbedding(nn.Module):
    """
    ### Embeddings for $t$
    """

    def __init__(self, n_channels: int):
        """
        * `n_channels` is the number of dimensions in the embedding
        """
        super().__init__()
        self.n_channels = n_channels
        # First linear layer
        self.lin1 = nn.Linear(self.n_channels // 4, self.n_channels)
        # Activation
        self.act = nn.GELU()
        # Second linear layer
        self.lin2 = nn.Linear(self.n_channels, self.n_channels)

    def forward(self, t: torch.Tensor):
        half_dim = self.n_channels // 8
        emb = math.log(10_000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device) * -emb)
        emb = t[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=1)

        # Transform with the MLP
        emb = self.act(self.lin1(emb))
        emb = self.lin2(emb)

        #
        return emb


def same_conv(in_channels: int, out_channels: int) -> nn.Conv2d:
    return nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1)


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

        self.time_embedding = TimeEmbedding(out_channels)

        self.convs = nn.Sequential(
            same_conv(in_channels, out_channels), nn.GELU(), same_conv(out_channels, out_channels), nn.GELU()
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> tuple[torch.Tensor]:
        t = self.time_embedding(t)
        x_res = self.convs(x)
        x_res += t.reshape(x_res.size(0), x_res.size(1), 1, 1)
        x = self.out(x_res)
        return x_res, x


class SmolUNet(nn.Module):
    def __init__(self, in_channels: int, out_channels: list[int] = [64, 128, 256, 512, 1024]) -> None:
        super().__init__()

        channels = [in_channels] + out_channels[:-1]

        self.downs = nn.ModuleList([UNetBlock(ic, oc, maxpool=True) for ic, oc in zip(channels[:-1], channels[1:])])
        self.latent_up = UNetBlock(out_channels[-2], out_channels[-1], upsample=True)
        self.ups = nn.ModuleList([UNetBlock(c * 2, c, upsample=c != out_channels[0]) for c in out_channels[::-1][1:]])
        self.out = same_conv(out_channels[0], in_channels)

    def forward(self, x: torch.Tensor, t: torch.Tensor):
        res = []
        for down in self.downs:
            x_res, x = down(x, t)
            res.append(x_res)

        _, x = self.latent_up(x, t)

        for xr, up in zip(res[::-1], self.ups):
            x = torch.cat([xr, x], dim=1)
            _, x = up(x, t)

        return self.out(x)


def main():
    smol_unet = SmolUNet(in_channels=1)
    x = torch.randn(8, 1, 32, 32)
    print(smol_unet(x).shape)


if __name__ == "__main__":
    main()

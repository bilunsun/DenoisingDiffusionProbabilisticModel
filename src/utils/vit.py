import math
from typing import Callable

import torch
from einops import rearrange
from einops.layers.torch import Rearrange
from torch import nn


class TimeEmbedding(nn.Module):
    def __init__(self, d_model: int) -> None:
        super().__init__()

        self.d_model = d_model

        self.proj = nn.Sequential(
            nn.Linear(self.d_model // 4, self.d_model), nn.GELU(), nn.Linear(self.d_model, self.d_model)
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half_dim = self.d_model // 8
        time_emb = math.log(10_000) / (half_dim - 1)
        time_emb = torch.exp(torch.arange(half_dim, device=t.device) * -time_emb)
        time_emb = t[:, None] * time_emb[None, :]
        time_emb = torch.cat((time_emb.sin(), time_emb.cos()), dim=1)

        time_emb = self.proj(time_emb)
        time_emb = time_emb.unsqueeze(1)

        return time_emb


class PreNorm(nn.Module):
    def __init__(self, d_model: int, fn: Callable) -> None:
        super().__init__()

        self.norm = nn.LayerNorm(d_model)
        self.fn = fn

    def forward(self, x: torch.Tensor, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, d_model: int, ffwd_dim: int, dropout: float):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(d_model, ffwd_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffwd_dim, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dim_head: int, dropout: float) -> None:
        super().__init__()

        inner_dim = dim_head * n_heads
        project_out = not (n_heads == 1 and dim_head == d_model)

        self.n_heads = n_heads
        self.scale = 1 / math.sqrt(dim_head)

        self.to_qkv = nn.Linear(d_model, inner_dim * 3, bias=False)  # 3 for {Q, K, V}
        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        self.to_out = (
            nn.Sequential(nn.Linear(inner_dim, d_model), nn.Dropout(dropout)) if project_out else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=self.n_heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, "b h n d -> b n (h d)")

        return self.to_out(out)


class Transformer(nn.Module):
    def __init__(self, d_model: int, depth: int, n_heads: int, dim_head: int, ffwd_dim: int, dropout: float) -> None:
        super().__init__()

        self.attn_blocks = nn.ModuleList([])
        self.ff_blocks = nn.ModuleList([])

        for _ in range(depth):
            self.attn_blocks.append(
                PreNorm(d_model, Attention(d_model, n_heads=n_heads, dim_head=dim_head, dropout=dropout))
            ),
            self.ff_blocks.append(PreNorm(d_model, FeedForward(d_model, ffwd_dim, dropout=dropout)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for attn, ff in zip(self.attn_blocks, self.ff_blocks):
            x = attn(x) + x
            x = ff(x) + x

        return x


class ViT(nn.Module):
    def __init__(
        self,
        image_size: int,
        patch_size: int,
        d_model: int,
        depth: int,
        n_heads: int,
        ffwd_dim: int,
        n_channels: int,
        dim_head: int,
        dropout: float,
    ):
        super().__init__()

        assert (
            image_size % patch_size == 0
        ), f"The given image_size={image_size} is not divisible by the given patch_size={patch_size}"

        self.num_patches = (image_size // patch_size) ** 2
        patch_dim = n_channels * patch_size**2

        self.to_patch_embedding = nn.Sequential(
            Rearrange("b c (h p1) (w p2) -> b (h w) (p1 p2 c)", p1=patch_size, p2=patch_size),
            nn.Linear(patch_dim, d_model),
        )

        self.time_embedding = TimeEmbedding(d_model)
        self.pos_embedding = nn.Parameter(torch.randn(1, self.num_patches, d_model))
        self.transformer = Transformer(d_model, depth, n_heads, dim_head, ffwd_dim, dropout)
        self.to_image = nn.Sequential(
            nn.Linear(d_model, patch_dim),
            Rearrange(
                "b (h w) (p1 p2 c) -> b c (h p1) (w p2)",
                c=n_channels,
                h=image_size // patch_size,
                w=image_size // patch_size,
                p1=patch_size,
                p2=patch_size,
            ),
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        x = self.to_patch_embedding(x)

        x += self.pos_embedding[:, : self.num_patches]
        x += self.time_embedding(t)

        x = self.transformer(x)
        x = self.to_image(x)

        return x


def main():
    image_size = 32
    patch_size = 8
    n_channels = 3
    x = torch.randn(1, n_channels, image_size, image_size)

    t = torch.randint(0, 200, size=(1,))
    vit = ViT(
        image_size=image_size,
        patch_size=patch_size,
        d_model=512,
        depth=6,
        n_heads=8,
        ffwd_dim=2048,
        n_channels=n_channels,
        dim_head=64,
        dropout=0.0,
    )

    with torch.no_grad():
        out = vit(x, t)
        print(out.shape)

    # import matplotlib.pyplot as plt

    # _, (left, right) = plt.subplots(1, 2)
    # left.imshow(rearrange(x, "b c h w -> h w (b c)"))
    # right.imshow(rearrange(out, "b c h w -> h w (b c)"))
    # plt.show()

    # assert torch.allclose(x, out, atol=1e-3)
    # print("ok")


if __name__ == "__main__":
    main()

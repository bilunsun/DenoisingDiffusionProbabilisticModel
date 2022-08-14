import matplotlib.pyplot as plt
import torch
from torchvision.utils import make_grid

from utils.data import (MNISTDataModule, get_inverse_transform,
                        get_mnist_dataset)
from utils.ddpm import Diffusion


def main():
    train_set, _ = get_mnist_dataset("data")
    noise = Diffusion(beta_range=[1e-4, 2e-2], T=200)
    x0, _ = train_set[9]

    t = torch.LongTensor([0, 20, 40, 60, 80, 100, 120, 140, 160, 199])
    xt = noise.q_sample(x0, t)
    inv_trans = get_inverse_transform()
    xt = inv_trans(xt)
    print(xt.min().item(), xt.max().item(), xt.mean().item(), xt.std().item())
    grid = make_grid(xt, nrow=len(t))

    plt.imshow(grid.permute(2, 1, 0).transpose(0, 1).numpy(), cmap="gray")
    plt.show()


if __name__ == "__main__":
    main()

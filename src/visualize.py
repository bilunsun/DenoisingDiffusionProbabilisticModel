import hydra
import matplotlib.pyplot as plt
import torch
from omegaconf import OmegaConf
from torchvision.utils import make_grid

from utils.data import get_inverse_transform


def main():
    schedule_config = OmegaConf.create(
        {
            "_target_": "utils.ddpm.Diffusion",
            "schedule_config": {"_target_": "utils.ddpm.Diffusion._cosine_beta_schedule", "T": 1000, "s": 0.008},
            # "schedule_config": {
            #     "_target_": "utils.ddpm.Diffusion._linear_beta_schedule",
            #     "beta_1": 1e-4,
            #     "beta_T": 2e-2,
            #     "T": 1000,
            # },
        }
    )
    dataset_config = OmegaConf.create(
        {
            "_target_": "utils.data.DataModule",
            "batch_size": 1,
            "dataset_config": {
                "_target_": "utils.data.get_celebA_dataset",
                "resize": 32,
                "data_dir": "data",
                "val_len": 50,
            },
        }
    )

    diffusion = hydra.utils.instantiate(schedule_config, _recursive_=False)
    dm = hydra.utils.instantiate(dataset_config, _recursive_=False)
    dm.setup()
    x0, _ = dm.train_set[9]

    plt.plot(diffusion.alpha_bar.numpy())
    plt.show()

    t = torch.linspace(0, diffusion.T - 1, 10).round().long()
    # t = torch.LongTensor([0, 20, 40, 60, 80, 100, 120, 140, 160, 199])
    xt = diffusion.q_sample(x0, t)
    inv_trans = get_inverse_transform()
    xt = inv_trans(xt)
    print(xt.min().item(), xt.max().item(), xt.mean().item(), xt.std().item())
    grid = make_grid(xt, nrow=len(t), scale_each=True, normalize=True)

    plt.imshow(grid.permute(2, 1, 0).transpose(0, 1).numpy())
    plt.show()


if __name__ == "__main__":
    main()

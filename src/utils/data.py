import os

import hydra
import pytorch_lightning as pl
from omegaconf import DictConfig
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torchvision.datasets import MNIST, CelebA, ImageFolder


def get_transform(resize: int):
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Resize(resize),
            transforms.CenterCrop(resize),
            transforms.Lambda(lambda t: (t * 2) - 1),
        ]
    )

    return transform


def get_inverse_transform():
    transform = transforms.Compose([transforms.Lambda(lambda t: (t + 1) / 2)])

    return transform


def get_mnist_dataset(data_dir: str, resize: int):
    transform = get_transform(resize)
    train_set = MNIST(data_dir, train=True, transform=transform, download=True)
    val_set = MNIST(data_dir, train=False, transform=transform, download=True)

    return train_set, val_set


def get_celebA_dataset(data_dir: str, resize: int, val_len: int):
    transform = get_transform(resize)
    dataset = CelebA(data_dir, split="all", transform=transform, download=True)
    train_len = len(dataset) - val_len
    train_set, val_set = random_split(dataset, [train_len, val_len])

    return train_set, val_set


def get_image_folder_dataset(data_dir: str, image_dir: str, resize: int, val_len: int):
    root_dir = os.path.join(data_dir, image_dir)
    transform = get_transform(resize)
    is_valid_file = lambda f: f.endswith(".jpg") or f.endswith(".jpeg")
    dataset = ImageFolder(root_dir, transform=transform, is_valid_file=is_valid_file)
    train_len = len(dataset) - val_len
    train_set, val_set = random_split(dataset, [train_len, val_len])

    return train_set, val_set


class DataModule(pl.LightningDataModule):
    def __init__(self, dataset_config: DictConfig, batch_size: int, shuffle: bool = True, num_workers: int = 1):
        super().__init__()

        self.dataset_config = dataset_config
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers

    def setup(self, stage=None):
        self.train_set, self.val_set = hydra.utils.instantiate(self.dataset_config)

    def train_dataloader(self):
        return DataLoader(
            self.train_set, batch_size=self.batch_size, shuffle=self.shuffle, num_workers=self.num_workers
        )

    def val_dataloader(self):
        return DataLoader(self.val_set, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)


def main():
    dm = DataModule(resize=32, batch_size=32, data_dir="data")
    dm.setup()
    for batch in dm.train_dataloader():
        print(batch[0].shape)
        break

    for batch in dm.val_dataloader():
        print(batch[0].shape)
        break


if __name__ == "__main__":
    main()

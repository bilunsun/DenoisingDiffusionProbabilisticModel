import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST


def get_transform(resize: int):
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Resize(resize), transforms.Lambda(lambda t: (t * 2) - 1)]
    )

    return transform


def get_inverse_transform():
    transform = transforms.Compose([transforms.Lambda(lambda t: (t + 1) / 2)])

    return transform


# TODO: Determine whether a val_set is needed
def get_mnist_dataset(data_dir: str, resize: int):
    assert resize == 32
    transform = get_transform(resize)
    train_set = MNIST(data_dir, train=True, transform=transform, download=True)
    val_set = MNIST(data_dir, train=False, transform=transform, download=True)

    return train_set, val_set


class MNISTDataModule(pl.LightningDataModule):
    def __init__(
        self, resize: int, batch_size: int, data_dir: str, shuffle: bool = True, num_workers: int = 1, **kwargs
    ):
        super().__init__()

        self.resize = resize
        self.batch_size = batch_size
        self.data_dir = data_dir
        self.shuffle = shuffle
        self.num_workers = num_workers

    def setup(self, stage=None):
        self.train_set, self.val_set = get_mnist_dataset(self.data_dir, self.resize)

    def train_dataloader(self):
        return DataLoader(
            self.train_set, batch_size=self.batch_size, shuffle=self.shuffle, num_workers=self.num_workers
        )

    def val_dataloader(self):
        return DataLoader(self.val_set, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)


def main():
    mnist_dm = MNISTDataModule(batch_size=32, data_dir="data")
    mnist_dm.setup()
    for batch in mnist_dm.train_dataloader():
        print(batch[0].shape)
        break

    for batch in mnist_dm.val_dataloader():
        print(batch[0].shape)
        break


if __name__ == "__main__":
    main()

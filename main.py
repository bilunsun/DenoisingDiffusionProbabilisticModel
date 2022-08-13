import hydra
from omegaconf import DictConfig


@hydra.main(config_path="conf", config_name="base.yaml", version_base=None)
def main(config: DictConfig):
    # Import inside main; see https://github.com/facebookresearch/hydra/issues/934
    from src.training_pipeline import train

    train(config)


if __name__ == "__main__":
    main()

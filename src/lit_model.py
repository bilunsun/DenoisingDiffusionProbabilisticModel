import hydra
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from omegaconf import DictConfig
from torchvision.utils import make_grid

from src.utils import get_logger
from src.utils.data import get_inverse_transform
from src.utils.ddpm import Diffusion, gather

log = get_logger(__name__)


class BaseModel(pl.LightningModule):
    def __init__(
        self,
        models_config: DictConfig,
        optimizer_config: DictConfig,
        scheduler_config: DictConfig,
        diffusion_config: DictConfig,
        use_weights_path: str,
    ) -> None:
        super().__init__()

        self.save_hyperparameters()

        # Instantiate a model with random weights, or load them from a checkpoint
        if self.hparams.use_weights_path is None:
            for model_name, model_config in self.hparams.models_config.items():
                model = hydra.utils.instantiate(model_config)
                setattr(self, model_name, model)
        else:
            ckpt = BaseModel.load_from_checkpoint(self.hparams.use_weights_path)

            for model_name in ckpt.hparams.models_config:
                model = getattr(ckpt, model_name)
                setattr(self, model_name, model)

        self.optimizer = hydra.utils.instantiate(self.hparams.optimizer_config, params=self.parameters())
        self.scheduler = (
            hydra.utils.instantiate(self.hparams.scheduler_config, optimizer=self.optimizer)
            if self.hparams.scheduler_config is not None
            else None
        )

        # Diffusion Configs
        self.diffusion = hydra.utils.instantiate(self.hparams.diffusion_config, _recursive_=False)

    def on_fit_start(self) -> None:
        self.diffusion = self.diffusion.to(self.device)

    def training_step(self, batch, _) -> torch.Tensor:
        x0, _ = batch
        t = self.diffusion.sample_t(size=(x0.size(0),))
        noise = torch.randn_like(x0)

        xt = self.diffusion.q_sample(x0, t, eps=noise)
        eps_theta = self.eps_model(xt, t)

        loss = F.mse_loss(noise, eps_theta)
        self.log("train_loss", loss, prog_bar=True, logger=True)

        return loss

    def validation_step(self, batch, _) -> None:
        x0, _ = batch
        t = self.diffusion.sample_t(size=(x0.size(0),))
        noise = torch.randn_like(x0)

        xt = self.diffusion.q_sample(x0, t, eps=noise)
        eps_theta = self.eps_model(xt, t)

        loss = F.mse_loss(noise, eps_theta)
        self.log("val_loss", loss, prog_bar=True, logger=True)

    def on_validation_epoch_end(self) -> None:
        xt = torch.randn(1, 3, 32, 32, device=self.device)
        saved_xt = [xt]
        # save_every = self.diffusion.T // 8
        save_indices = torch.linspace(0, self.diffusion.T, 10).round().long()
        for i in reversed(range(self.diffusion.T - 1)):
            t = torch.tensor([i], device=self.device).long()
            xt = self.p_sample(xt, t)

            if i in save_indices:
                saved_xt.append(xt)

        vis_xt = torch.cat(saved_xt)
        inv_trans = get_inverse_transform()
        vis_xt = inv_trans(vis_xt)

        grid = make_grid(vis_xt, nrow=len(vis_xt), scale_each=True, normalize=True)
        self.logger.log_image(key="samples", images=[grid])

    def configure_optimizers(self):
        if self.scheduler is None:
            return self.optimizer

        return [self.optimizer], [{"scheduler": self.scheduler, "interval": "epoch"}]

    def p_sample(self, xt: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        eps_theta = self.eps_model(xt, t)
        return self.diffusion.p_sample(xt, t, eps_theta)

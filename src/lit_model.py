import hydra
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from typing import Optional

from src.utils import get_logger
from src.utils.ddpm import gather


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
        # self.scheduler = hydra.utils.instantiate(self.hparams.scheduler_config, optimizer=self.optimizer)

        # Diffusion Configs
        self.register_buffer(
            name="beta",
            tensor=torch.linspace(
                start=self.hparams.diffusion_config.beta_1,
                end=self.hparams.diffusion_config.beta_T,
                steps=self.hparams.diffusion_config.T,
            ),
        )
        self.register_buffer(name="alpha", tensor=1.0 - self.beta)
        self.register_buffer(name="alpha_bar", tensor=torch.cumprod(input=self.alpha, dim=0))
        self.T = self.hparams.diffusion_config.T

    def on_fit_start(self) -> None:
        self.beta = self.beta.to(self.device)
        self.alpha = self.alpha.to(self.device)
        self.alpha_bar = self.alpha_bar.to(self.device)

    def training_step(self, batch, _) -> torch.Tensor:
        x, _ = batch
        t = torch.randint(low=0, high=self.T, size=(x.size(0),)).to(self.device)

        noise = torch.randn_like(x)

        xt = self.q_sample(x, t, eps=noise)
        eps_theta = self.eps_model(xt, t)

        loss = F.mse_loss(noise, eps_theta)
        self.log("train_loss", loss, prog_bar=True, logger=True)

        return loss

    def validation_step(self, batch, _) -> None:
        x, _ = batch
        t = torch.randint(low=0, high=self.T, size=(x.size(0),)).to(self.device)
        noise = torch.randn_like(x).to(self.device)

        xt = self.q_sample(x, t, eps=noise)
        eps_theta = self.eps_model(xt, t)

        loss = F.mse_loss(noise, eps_theta)
        self.log("val_loss", loss, prog_bar=True, logger=True)

    def configure_optimizers(self):
        return self.optimizer

    def q_xt_x0(self, x0: torch.Tensor, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        gathered = gather(v=self.alpha_bar, index=t)
        mean = gathered**0.5 * x0
        var = 1 - gathered

        return mean, var

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, eps: Optional[torch.Tensor] = None) -> torch.Tensor:
        if eps is None:
            eps = torch.randn_like(x0).to(self.device)

        mean, var = self.q_xt_x0(x0, t)

        return mean + (var**0.5) * eps

    def p_sample(self, xt: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        eps_theta = self.eps_model(xt, t)
        alpha_bar = gather(v=self.alpha, index=t)
        eps_coeff = (1 - alpha) / (1 - alpha_bar) ** 0.5
        mean = 1 / (alpha**0.5) * (xt - eps_coeff * eps_theta)
        var = gather(v=self.beta, index=t)
        eps = torch.randn(xt.shape).to(self.device)

        return mean + (var**0.5) * eps

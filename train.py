import os

import torch
import pytorch_lightning as pl
import yaml
import time
from pytorch_lightning.cli import LightningCLI

from climax.global_forecast.datamodule import GlobalForecastDataModule
from climax.global_forecast.module import GlobalForecastModule
from climax.regional_forecast.datamodule import RegionalForecastDataModule
from climax.regional_forecast.module import RegionalForecastModule
from climax.climate_projection.module import ClimateProjectionModule
from climax.climate_projection.datamodule import ClimateBenchDataModule

from pytorch_lightning.profiler import PyTorchProfiler
from pytorch_lightning.loggers import TensorBoardLogger


cli = LightningCLI(
        model_class=RegionalForecastModule,
        datamodule_class=RegionalForecastDataModule,
        seed_everything_default=42,
        save_config_overwrite=True,
        run=False,
        auto_registry=True,
        parser_kwargs={"parser_mode": "omegaconf", "error_handler": None},
    )
os.makedirs(cli.trainer.default_root_dir, exist_ok=True)

cli.datamodule.set_patch_size(cli.model.get_patch_size())  # only for regional forecast

normalization = cli.datamodule.output_transforms
mean_norm, std_norm = normalization.mean, normalization.std
mean_denorm, std_denorm = -mean_norm / std_norm, 1 / std_norm
cli.model.set_denormalization(mean_denorm, std_denorm)
cli.model.set_lat_lon(*cli.datamodule.get_lat_lon())
cli.model.set_pred_range(cli.datamodule.hparams.predict_range)
cli.model.set_val_clim(cli.datamodule.val_clim)
cli.model.set_test_clim(cli.datamodule.test_clim)


# logger = TensorBoardLogger(f'./tensorboard', name='regional_forecast')

trainer = pl.Trainer(
    accelerator='gpu',
    devices=[2],
    max_epochs=50,
    # limit_train_batches=10,
    # limit_val_batches=0,
    # limit_test_batches=10,
    # logger=logger,
    precision=16,
    inference_mode=True,
)

trainer.fit(cli.model, datamodule=cli.datamodule)

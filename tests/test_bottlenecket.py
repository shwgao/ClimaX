# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
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
from have_more_fun import trace_handler
from have_fun import start_record_memory_history, export_memory_snapshot, stop_record_memory_history


# profile_task = f'global11_forecast_climax_test-{time.strftime("%m%d%H%M")}'


def main_profiler():
    profile_task = f'regional_forecast_climax_test-{time.strftime("%m%d%H%M")}'

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

    # logger = TensorBoardLogger(f'./profile/{profile_task}', name=profile_task)
    profiler = PyTorchProfiler(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        schedule=torch.profiler.schedule(wait=2, warmup=1, active=2, repeat=1),
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
        with_modules=True,
        # on_trace_ready=trace_handler,
        on_trace_ready=torch.profiler.tensorboard_trace_handler(f'./profile/{profile_task}'),
    )
    trainer = pl.Trainer(
        accelerator='gpu',
        devices=[1],
        max_epochs=1,
        limit_train_batches=5,
        limit_val_batches=5,
        limit_test_batches=5,
        # logger=logger,
        # profiler=profiler,
        precision=16,
        inference_mode=True,
    )

    # start_record_memory_history()

    # trainer.fit(cli.model, datamodule=cli.datamodule)
    # trainer.test(cli.model, datamodule=cli.datamodule)

    # Create the memory snapshot file
    # export_memory_snapshot()

    # Stop recording memory snapshot history
    # stop_record_memory_history()

    # trainer.fit(cli.model, datamodule=cli.datamodule)
    
    # cli.model.eval()
    cli.datamodule.setup()
    test_loader = cli.datamodule.test_dataloader()
    x, y, lead_times, variables, out_variables, region_info = next(iter(test_loader))
    
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
        schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=1),
        on_trace_ready=torch.profiler.tensorboard_trace_handler(f'./profile/{profile_task}'),
        record_shapes=True,
        profile_memory=True,
        with_stack=True
    ) as prof:
        # for step, batch_data in enumerate(test_loader):
        #     prof.step()  # Need to call this at each step to notify profiler of steps' boundary.
        #     if step >= 1 + 1 + 3:
        #         break
        #     cli.model.test_step(batch_data, step)
        trainer.fit(cli.model, datamodule=cli.datamodule)
    
    example_input = (x, y, lead_times, variables, out_variables, None, torch.tensor(cli.model.lat), region_info, {})
    
    cli.model.net.to_onnx('./checkpoints/regional_forecast.onnx', input_sample=example_input, export_params=True)


def main_profiler_global():
    profile_task = f'global11_forecast_climax_test-{time.strftime("%m%d%H%M")}'

    cli = LightningCLI(
        model_class=GlobalForecastModule,
        datamodule_class=GlobalForecastDataModule,
        seed_everything_default=42,
        save_config_overwrite=True,
        run=False, 
        auto_registry=True,
        parser_kwargs={"parser_mode": "omegaconf", "error_handler": None},
    )
    os.makedirs(cli.trainer.default_root_dir, exist_ok=True)

    normalization = cli.datamodule.output_transforms
    mean_norm, std_norm = normalization.mean, normalization.std
    mean_denorm, std_denorm = -mean_norm / std_norm, 1 / std_norm
    cli.model.set_denormalization(mean_denorm, std_denorm)
    cli.model.set_lat_lon(*cli.datamodule.get_lat_lon())
    cli.model.set_pred_range(cli.datamodule.hparams.predict_range)
    cli.model.set_val_clim(cli.datamodule.val_clim)
    cli.model.set_test_clim(cli.datamodule.test_clim)

    # # logger = TensorBoardLogger(f'./profile/{profile_task}', name=profile_task)
    # profiler = PyTorchProfiler(
    #     activities=[
    #         torch.profiler.ProfilerActivity.CPU,
    #         torch.profiler.ProfilerActivity.CUDA,
    #     ],
    #     schedule=torch.profiler.schedule(wait=2, warmup=1, active=2, repeat=1),
    #     record_shapes=True,
    #     profile_memory=True,
    #     with_stack=True,
    #     with_modules=True,
    #     # on_trace_ready=trace_handler,
    #     on_trace_ready=torch.profiler.tensorboard_trace_handler(f'./profile/{profile_task}'),
    # )
    trainer = pl.Trainer(
        accelerator='gpu',
        devices=[1],
        max_epochs=1,
        limit_train_batches=5,
        limit_val_batches=5,
        limit_test_batches=5,
        # logger=logger,
        # profiler=profiler,
        precision=16,
        fast_dev_run=True,
        inference_mode=True,
    )

    # trainer.fit(cli.model, datamodule=cli.datamodule)
    
    cli.model.eval()
    cli.datamodule.setup()
    test_loader = cli.datamodule.test_dataloader()
    x, y, lead_times, variables, out_variables = next(iter(test_loader))
    
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
        schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=1),
        on_trace_ready=torch.profiler.tensorboard_trace_handler(f'./profile/{profile_task}'),
        record_shapes=True,
        profile_memory=True,
        with_stack=True
    ) as prof:
        # for step, batch_data in enumerate(test_loader):
        #     prof.step()  # Need to call this at each step to notify profiler of steps' boundary.
        #     if step >= 1 + 1 + 3:
        #         break
        #     cli.model.test_step(batch_data, step)
        trainer.test(cli.model, datamodule=cli.datamodule)
    
    example_input = (x, y, lead_times, variables, out_variables, None, None)
    
    cli.model.net.to_onnx('./checkpoints/global_forecast.onnx', input_sample=tuple(example_input), export_params=True)



def main_profiler_projection():
    profile_task = f'projection_bz4_forecast_climax_test-{time.strftime("%m%d%H%M")}'

    # Initialize Lightning with the model and data modules, and instruct it to parse the config yml
    cli = LightningCLI(
        model_class=ClimateProjectionModule,
        datamodule_class=ClimateBenchDataModule,
        seed_everything_default=42,
        save_config_overwrite=True,
        run=False,
        # auto_registry=True,
        parser_kwargs={"parser_mode": "omegaconf", "error_handler": None},
    )
    os.makedirs(cli.trainer.default_root_dir, exist_ok=True)

    normalization = cli.datamodule.dataset_train.out_transform
    mean_norm, std_norm = normalization.mean, normalization.std
    mean_denorm, std_denorm = -mean_norm / std_norm, 1 / std_norm
    cli.model.set_denormalization(mean_denorm, std_denorm)
    cli.model.set_lat_lon(*cli.datamodule.get_lat_lon())
    cli.model.set_pred_range(0)
    cli.model.set_val_clim(None)
    cli.model.set_test_clim(cli.datamodule.get_test_clim())

    # logger = TensorBoardLogger(f'./profile/{profile_task}', name=profile_task)
    profiler = PyTorchProfiler(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        schedule=torch.profiler.schedule(wait=2, warmup=1, active=2, repeat=1),
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
        with_modules=True,
        # on_trace_ready=trace_handler,
        on_trace_ready=torch.profiler.tensorboard_trace_handler(f'./profile/{profile_task}'),
    )
    
    trainer = pl.Trainer(
        accelerator='gpu',
        devices=[2],
        max_epochs=1,
        limit_train_batches=5,
        limit_val_batches=5,
        limit_test_batches=5,
        # logger=logger,
        # profiler=profiler,
        precision=16,
        inference_mode=True,
    )

    # trainer.fit(cli.model, datamodule=cli.datamodule)
    
    test_loader = cli.datamodule.test_dataloader()
    x, y, lead_times, variables, out_variables = next(iter(test_loader))
    
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
        schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=1),
        on_trace_ready=torch.profiler.tensorboard_trace_handler(f'./profile/{profile_task}'),
        record_shapes=True,
        profile_memory=True,
        with_stack=True
    ) as prof:
        # for step, batch_data in enumerate(test_loader):
        #     prof.step()  # Need to call this at each step to notify profiler of steps' boundary.
        #     if step >= 1 + 1 + 3:
        #         break
        #     cli.model.test_step(batch_data, step)
        trainer.fit(cli.model, datamodule=cli.datamodule)
    
    example_input = (x, y, lead_times, variables, out_variables, None, None)
    
    cli.model.net.to_onnx('./checkpoints/projection.onnx', input_sample=example_input, export_params=True)


if __name__ == "__main__":
    # main()
    # main_profiler()
    # main_profiler_global()
    main_profiler_projection()

# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import os
import sys

import torch
import pytorch_lightning as pl
import yaml
import time
from pytorch_lightning.cli import LightningCLI

from climax.arch import ClimaX

import torch.nn.functional as F
import torch.nn as nn

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

# enable the flash attention backend
# torch.backends.cuda.enable_flash_sdp(False)
# torch.backends.cuda.enable_math_sdp(True)
# torch.backends.cuda.enable_mem_efficient_sdp(False)

from torch.nn.attention import SDPBackend, sdpa_kernel

# profile_task = f'global11_forecast_climax_test-{time.strftime("%m%d%H%M")}'

class ModelConfigGlobal:
    def __init__(self, 
                 default_vars=None, 
                 img_size=None, 
                 patch_size=2, 
                 embed_dim=1024, 
                 depth=8, 
                 decoder_depth=2, 
                 num_heads=16, 
                 mlp_ratio=4, 
                 drop_path=0.1, 
                 drop_rate=0.1):
        if default_vars is None:
            default_vars = [
                "land_sea_mask", "orography", "lattitude", "2m_temperature",
                "10m_u_component_of_wind", "10m_v_component_of_wind", "geopotential_50",
                "geopotential_250", "geopotential_500", "geopotential_600", "geopotential_700",
                "geopotential_850", "geopotential_925", "u_component_of_wind_50",
                "u_component_of_wind_250", "u_component_of_wind_500", "u_component_of_wind_600",
                "u_component_of_wind_700", "u_component_of_wind_850", "u_component_of_wind_925",
                "v_component_of_wind_50", "v_component_of_wind_250", "v_component_of_wind_500",
                "v_component_of_wind_600", "v_component_of_wind_700", "v_component_of_wind_850",
                "v_component_of_wind_925", "temperature_50", "temperature_250", "temperature_500",
                "temperature_600", "temperature_700", "temperature_850", "temperature_925",
                "relative_humidity_50", "relative_humidity_250", "relative_humidity_500",
                "relative_humidity_600", "relative_humidity_700", "relative_humidity_850",
                "relative_humidity_925", "specific_humidity_50", "specific_humidity_250",
                "specific_humidity_500", "specific_humidity_600", "specific_humidity_700",
                "specific_humidity_850", "specific_humidity_925"
            ]
        if img_size is None:
            img_size = [32, 64]
        
        self.out_variables = ["geopotential_500", "temperature_850", "2m_temperature", 
                              "10m_u_component_of_wind", "10m_v_component_of_wind"]
        self.predict_range: 72
        
        self.default_vars = default_vars
        self.img_size = img_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.depth = depth
        self.decoder_depth = decoder_depth
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.drop_path = drop_path
        self.drop_rate = drop_rate


class CausalSelfAttention(nn.Module):

    def __init__(self, num_heads: int, embed_dimension: int, bias: bool=False, is_causal: bool=False, dropout:float=0.0):
        super().__init__()
        assert embed_dimension % num_heads == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(embed_dimension, 3 * embed_dimension, bias=bias)
        # output projection
        self.c_proj = nn.Linear(embed_dimension, embed_dimension, bias=bias)
        # regularization
        self.dropout = dropout
        self.resid_dropout = nn.Dropout(dropout)
        self.num_heads = num_heads
        self.embed_dimension = embed_dimension
        # Perform causal masking
        self.is_causal = is_causal

    def forward(self, x):
        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        query_projected = self.c_attn(x)

        batch_size = query_projected.size(0)
        embed_dim = query_projected.size(2)
        head_dim = embed_dim // (self.num_heads * 3)

        query, key, value = query_projected.chunk(3, -1)
        query = query.view(batch_size, -1, self.num_heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, self.num_heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, self.num_heads, head_dim).transpose(1, 2)

        if self.training:
            dropout = self.dropout
            is_causal = self.is_causal
        else:
            dropout = 0.0
            is_causal = False

        y = F.scaled_dot_product_attention(query, key, value, attn_mask=None, dropout_p=dropout, is_causal=is_causal)
        y = y.transpose(1, 2).view(batch_size, -1, self.num_heads * head_dim)

        y = self.resid_dropout(self.c_proj(y))
        return y


num_heads = 8
heads_per_dim = 64
embed_dimension = num_heads * heads_per_dim
dtype = torch.float16
model = CausalSelfAttention(num_heads=num_heads, embed_dimension=embed_dimension, bias=False, is_causal=True, dropout=0.1).to("cuda").to(dtype).eval()
print(model)


def main_profiler_regional():
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
    
    profile_task = f'regional_bz40_inference_SouthAmerica-{time.strftime("%m%d%H%M")}-fused'

    # logger = TensorBoardLogger(f'./profile/{profile_task}', name=profile_task)
    profiler = PyTorchProfiler(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        schedule=torch.profiler.schedule(wait=4, warmup=2, active=4, repeat=1),
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
        with_modules=True,
        # on_trace_ready=trace_handler,
        on_trace_ready=torch.profiler.tensorboard_trace_handler(f'./profile/offload_debug/{profile_task}'),
    )
    trainer = pl.Trainer(
        accelerator='gpu',
        devices=[3],
        max_epochs=1,
        limit_train_batches=10,
        limit_val_batches=0,
        limit_test_batches=10,
        # logger=logger,
        profiler=profiler,
        precision=16,
        inference_mode=True,
    )

    trainer.test(cli.model, datamodule=cli.datamodule)
    
    def export_onnx():
        cli.datamodule.setup()
        test_loader = cli.datamodule.test_dataloader()
        x, y, lead_times, variables, out_variables, region_info = next(iter(test_loader))
        example_input = (x, y, lead_times, variables, out_variables, None, torch.tensor(cli.model.lat), region_info, {})
        cli.model.net.to_onnx('./checkpoints/regional_forecast.onnx', input_sample=example_input, export_params=True)


def main_profiler_global():
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

    profile_task = f'global_bz20_inference-{time.strftime("%m%d%H%M")}-flash'
    # # logger = TensorBoardLogger(f'./profile/{profile_task}', name=profile_task)
    profiler = PyTorchProfiler( activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=1),
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
        with_modules=True,
        # on_trace_ready=trace_handler,
        on_trace_ready=torch.profiler.tensorboard_trace_handler(f'./profile/climax/{profile_task}'),
    )
    
    trainer = pl.Trainer(
        accelerator='gpu',
        devices=[3],
        max_epochs=1,
        limit_train_batches=5,
        limit_val_batches=0,
        limit_test_batches=5,
        # logger=logger,
        profiler=profiler,
        precision=32,
        inference_mode=True,
    )
    
    # cli.model = torch.compile(cli.model, backend='hidet')

    trainer.test(cli.model, datamodule=cli.datamodule)
    
    def export_onnx():
        cli.model.eval()
        cli.datamodule.setup()
        test_loader = cli.datamodule.test_dataloader()
        x, y, lead_times, variables, out_variables = next(iter(test_loader))
        example_input = (x, y, lead_times, variables, out_variables, None, None)
        cli.model.net.to_onnx('./checkpoints/global_forecast.onnx', input_sample=tuple(example_input), export_params=True)


def main_profiler_projection():
    
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
    cli.model.net = torch.compile(cli.model.net)
    
    profile_task = f'projection_bz12-train_compile-{time.strftime("%m%d%H%M")}'

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
        limit_val_batches=0,
        limit_test_batches=5,
        # logger=logger,
        # profiler=profiler,
        precision=16,
        inference_mode=True,
    )

    trainer.fit(cli.model, datamodule=cli.datamodule)

    def export_onnx():
        test_loader = cli.datamodule.test_dataloader()
        x, y, lead_times, variables, out_variables = next(iter(test_loader))
        example_input = (x, y, lead_times, variables, out_variables, None, None)
        cli.model.net.to_onnx('./checkpoints/projection.onnx', input_sample=example_input, export_params=True)


def profile_straight():
    model_config = ModelConfigGlobal()
    
    net = ClimaX(
        default_vars=model_config.default_vars,
        img_size=model_config.img_size,
        patch_size=model_config.patch_size,
        embed_dim=model_config.embed_dim,
        depth=model_config.depth,
        decoder_depth=model_config.decoder_depth,
        num_heads=model_config.num_heads,
        mlp_ratio=model_config.mlp_ratio,
        drop_path=model_config.drop_path,
        drop_rate=model_config.drop_rate,
    )
    
    device = torch.device("cuda:0")
    net = net.to(device)
    
    batch = 40
    
    x = torch.randn(batch, 48, 32, 64, dtype=torch.float32).cuda()
    y = None
    lead_times = torch.tensor([72]*batch, dtype=torch.float32).cuda()
    variables = model_config.default_vars
    out_variables = model_config.out_variables
    
    net.eval()
    step = 0
    profile_task = f'global_bz{batch}_inference-{time.strftime("%m%d%H%M")}-oremptycache'
    
    with torch.no_grad():
        with sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION):
            with torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA,
                ],
                schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=1),
                record_shapes=True,
                profile_memory=True,
                with_stack=True,
                with_modules=True,
                on_trace_ready=torch.profiler.tensorboard_trace_handler(f'./profile/climax/{profile_task}'),
            ) as p:
                for i in range(10):
                    p.step()
                    if step >=5:
                        break
                    results = net(x, y, lead_times, variables, out_variables, None, None)
                    
        torch.cuda.memory._record_memory_history()
        
        results = net(x, y, lead_times, variables, out_variables, None, None)

        torch.cuda.memory._dump_snapshot("./profile/climax/my_snapshot.pickle")


def profile_mha():
    q = torch.randn(5120, 1, 1024).cuda()
    x = torch.randn(5120, 48, 1024).cuda()
    
    mha = torch.nn.MultiheadAttention(1024, 16, batch_first=True).cuda()
    
    mha = CausalSelfAttention(16, 1024).cuda()
    
    import torch.utils.benchmark as benchmark
    def benchmark_torch_function_in_microseconds(f, *args, **kwargs):
        t0 = benchmark.Timer(
            stmt="f(*args, **kwargs)", globals={"args": args, "kwargs": kwargs, "f": f}
        )
        return t0.blocked_autorange().mean * 1e6

    torch.cuda.synchronize()
    with sdpa_kernel(SDPBackend.MATH):
        math_time=benchmark_torch_function_in_microseconds(mha, x)
        torch.cuda.synchronize()
        print(f"The math implementation runs in {math_time:.3f} microseconds")
    
    with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
        math_time=benchmark_torch_function_in_microseconds(mha, x)
        torch.cuda.synchronize()
        print(f"The flash attention implementation runs in {math_time:.3f} microseconds")
    
    with sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION):
        math_time=benchmark_torch_function_in_microseconds(mha, x)
        torch.cuda.synchronize()
        print(f"The EFFICIENT_ATTENTION implementation runs in {math_time:.3f} microseconds")


def pytorch_tutorial():
        # Lets define a helpful benchmarking function:
    import torch.utils.benchmark as benchmark
    import torch.nn.functional as F
    def benchmark_torch_function_in_microseconds(f, *args, **kwargs):
        t0 = benchmark.Timer(
            stmt="f(*args, **kwargs)", globals={"args": args, "kwargs": kwargs, "f": f}
        )
        return t0.blocked_autorange().mean * 1e6

    device = torch.device("cuda:0")

    # Lets define the hyper-parameters of our input
    batch_size = 32
    max_sequence_len = 1024
    num_heads = 32
    embed_dimension = 32

    dtype = torch.float16

    query = torch.rand(batch_size, num_heads, max_sequence_len, embed_dimension, device=device, dtype=dtype)
    key = torch.rand(batch_size, num_heads, max_sequence_len, embed_dimension, device=device, dtype=dtype)
    value = torch.rand(batch_size, num_heads, max_sequence_len, embed_dimension, device=device, dtype=dtype)

    torch.cuda.reset_peak_memory_stats()
    print(f"The default implementation runs in {benchmark_torch_function_in_microseconds(F.scaled_dot_product_attention, query, key, value):.3f} microseconds")
    max_memory = torch.cuda.max_memory_allocated()
    print(f"Maximum memory allocated during execution: {max_memory / (1024 ** 2):.2f} MB")
    
    # Lets explore the speed of each of the 3 implementations
    from torch.nn.attention import SDPBackend, sdpa_kernel


    with sdpa_kernel(SDPBackend.MATH):
        torch.cuda.reset_peak_memory_stats()
        math_time=benchmark_torch_function_in_microseconds(F.scaled_dot_product_attention, query, key, value)
        max_memory = torch.cuda.max_memory_allocated()
        print(f"The math implementation runs in {math_time:.3f} microseconds")
        print(f"Maximum memory allocated during execution: {max_memory / (1024 ** 2):.2f} MB")

    with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
        
        torch.cuda.reset_peak_memory_stats()
        try:
            flash_time=benchmark_torch_function_in_microseconds(F.scaled_dot_product_attention, query, key, value)
            max_memory = torch.cuda.max_memory_allocated()
            print(f"The flash attention implementation runs in {flash_time:.3f} microseconds")
            print(f"Maximum memory allocated during execution: {max_memory / (1024 ** 2):.2f} MB")
        except RuntimeError:
            print("FlashAttention is not supported. See warnings for reasons.")

    with sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION):
        # measure the maximum memory usage
        
        torch.cuda.reset_peak_memory_stats()
        try:
            efficient_time=benchmark_torch_function_in_microseconds(F.scaled_dot_product_attention, query, key, value)
            max_memory = torch.cuda.max_memory_allocated()
            print(f"The memory efficient implementation runs in {efficient_time:.3f} microseconds")
            print(f"Maximum memory allocated during execution: {max_memory / (1024 ** 2):.2f} MB")
        except RuntimeError:
            print("EfficientAttention is not supported. See warnings for reasons.")

if __name__ == "__main__":
    # main()
    # main_profiler_regional()
    # main_profiler_global()
    # main_profiler_projection()
    profile_straight()
    # profile_mha()
    # pytorch_tutorial()
    # print(torch.backends.cuda.flash_sdp_enabled())
    # # True
    # print(torch.backends.cuda.mem_efficient_sdp_enabled())
    # # True
    # print(torch.backends.cuda.math_sdp_enabled())

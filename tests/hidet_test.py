import torch
import yaml
import time

from climax.arch import ClimaX
from climax.global_forecast.datamodule import GlobalForecastDataModule

import torch._dynamo
torch._dynamo.config.suppress_errors = True

# read init args from yaml
with open("configs/global_forecast_climax_test.yaml", "r") as f:
    args = yaml.load(f, Loader=yaml.FullLoader)

init_args = args['model']['net']['init_args']
data_module_args = args['data']

model = ClimaX(default_vars=init_args['default_vars'])
model_opt = torch.compile(model, backend='hidet')

data_module = GlobalForecastDataModule(root_dir=data_module_args['root_dir'], 
                                       variables=data_module_args['variables'], 
                                       buffer_size=data_module_args['buffer_size'], 
                                       out_variables=data_module_args['out_variables'], 
                                       predict_range=data_module_args['predict_range'], 
                                       batch_size=data_module_args['batch_size'], 
                                       num_workers=data_module_args['num_workers'], 
                                       pin_memory=data_module_args['pin_memory'])

data_module.setup()

profile_task = f'global_bz20_inference-{time.strftime("%m%d%H%M")}-or'

# 将数据移动到GPU（如果可用）
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

step = 0
# 运行模型
# model_opt.to(device)  # 确保模型在正确的设备上
# model_opt.eval()  # 设置模型为评估模式
model.to(device)
model.eval()

with torch.no_grad():
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
        on_trace_ready=torch.profiler.tensorboard_trace_handler(f'./profile/{profile_task}'),
    ) as p:
        try:
            for batch in data_module.test_dataloader():
                p.step()
                if step >=5:
                    break
                
                x, y, lead_times, variables, out_variables = batch
                
                x = x.to(device)
                y = y.to(device)
                lead_times = lead_times.to(device)
                
                model(x, y, lead_times, variables, out_variables, None, None)                
                
        except Exception as e:
            print(f"An error occurred during profiling: {e}")
        
        finally:
            # 确保分析器正确关闭
            p.stop()


# if __name__ == "__main__":
#     test_parallel_patch_embed()

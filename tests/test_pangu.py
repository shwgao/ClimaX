import torch
import yaml
import time

from pangu_weather.model import WeatherModel

# read init args from yaml
batch_size = 4


model = WeatherModel(192, [2,6,6,2], [6,12,12,6], 4, batch_size)

profile_task = f'bz32_inference-{time.strftime("%m%d%H%M")}-or'

# 将数据移动到GPU（如果可用）
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
datas = []
B = batch_size  # You can change this to your desired batch size
Z = 13
H = 1440
W = 721
C_upper = 5
C_surface = 4


for i in range(5):
    data = []
    # Generate random tensor for upper-air variables
    upper_air = torch.rand(B, Z, H, W, C_upper)
    # Generate random tensor for surface variables
    surface = torch.rand(B, H, W, C_surface)
    data.append(upper_air)
    data.append(surface)
    datas.append(data)

step = 0
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
        on_trace_ready=torch.profiler.tensorboard_trace_handler(f'./profile/pangu/{profile_task}'),
    ) as p:
        try:
            for batch in datas:
                p.step()
                if step >=5:
                    break
                                
                batch[0] = batch[0].to(device)
                batch[1] = batch[1].to(device)
                lead_times = lead_times.to(device)
                
                model(batch)                
                
        except Exception as e:
            print(f"An error occurred during profiling: {e}")
        
        finally:
            # 确保分析器正确关闭
            p.stop()


# if __name__ == "__main__":
#     test_parallel_patch_embed()

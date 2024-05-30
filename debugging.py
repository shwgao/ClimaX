import torch
from src.climax.arch import ClimaX


if __name__ == '__main__':
    # test ClimaX model
    model = ClimaX(default_vars=["t2m", "mslp", "u10", "v10"])
    x = torch.randn(2, 4, 32, 64)
    y = torch.randn(2, 4, 32, 64)
    lead_times = torch.randint(0, 24, (2,))
    variables = ["t2m", "mslp", "u10", "v10"]
    out_variables = ["t2m", "mslp", "u10", "v10"]
    metric = None
    lat = torch.randn(32, 64)
    loss, preds = model(x, y, lead_times, variables, out_variables, metric, lat)
    print(loss)
import torch, torch.nn as nn
class LatentPredictor(nn.Module):
    def __init__(self, dim: int = 768, hidden: int = 1024, depth: int = 2):
        super().__init__()
        layers, d = [], dim
        for _ in range(depth):
            layers += [nn.Linear(d, hidden), nn.GELU()]
            d = hidden
        layers += [nn.Linear(d, dim)]
        self.net = nn.Sequential(*layers)
    def forward(self, ctx_z: torch.Tensor) -> torch.Tensor:
        return self.net(ctx_z)

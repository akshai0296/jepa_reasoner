from dataclasses import dataclass
from typing import Dict
import torch
@dataclass
class ForwardOutput:
    ctx_z: torch.Tensor
    tgt_z: torch.Tensor
    pred_z: torch.Tensor
    losses: Dict[str, torch.Tensor]
class IJEPAModel:
    def __init__(self, ctx_enc, tgt_enc, predictor, decoders: Dict[str, object]):
        self.ctx_enc, self.tgt_enc = ctx_enc, tgt_enc
        self.predictor = predictor
        self.decoders = decoders
    def forward(self, context, target, domain: str) -> ForwardOutput:
        raise NotImplementedError

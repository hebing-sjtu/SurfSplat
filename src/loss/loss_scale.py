from dataclasses import dataclass

import torch
from jaxtyping import Float
from torch import Tensor

from ..dataset.types import BatchedExample
from ..model.decoder.decoder import DecoderOutput
from ..model.types import Gaussians
from .loss import Loss


@dataclass
class LossScaleCfg:
    weight: float


@dataclass
class LossScaleCfgWrapper:
    scale: LossScaleCfg


class LossScale(Loss[LossScaleCfg, LossScaleCfgWrapper]):
    def forward(
        self,
        prediction: DecoderOutput,
        batch: BatchedExample,
        gaussians: Gaussians,
        global_step: int,
        valid_depth_mask: Tensor | None
    ) -> Float[Tensor, ""]:
        
        scl_res = gaussians.scales_mul.abs().contiguous().view(-1)
        
        greater_than_one = scl_res[scl_res > 1] - 1  
        less_than_one = 1/scl_res[scl_res < 1] - 1
        
        loss = torch.cat([greater_than_one ** 2, less_than_one ** 2])
        
        return self.cfg.weight * loss.mean()
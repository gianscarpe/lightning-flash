# Copyright The PyTorch Lightning team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import torch.nn as nn

from typing import List
from attrdict import AttrDict

from flash.core.registry import FlashRegistry
from flash.core.utilities.imports import _VISSL_AVAILABLE

if _VISSL_AVAILABLE:
    from vissl.models.heads import MODEL_HEADS_REGISTRY


def swav_head(
    dims: List[int],
    use_bn: bool,
    num_clusters: int,
    use_bias: bool = True,
    return_embeddings: bool = True,
    skip_last_bn: bool = True,
    normalize_feats: bool = True,
    activation_name: str = "ReLU",
    use_weight_norm_prototypes: bool = False,
    batchnorm_eps: float = 1e-5,
    batchnorm_momentum: float = 0.1,
    **kwargs,
) -> nn.Module:
    cfg = {
        "model_config": AttrDict({
            "HEAD": AttrDict({"BATCHNORM_EPS": batchnorm_eps, "BATCHNORM_MOMENTUM": batchnorm_momentum})
        }),
        "dims": dims,
        "use_bn": use_bn,
        "num_clusters": num_clusters,
        "use_bias": use_bias,
        "return_embeddings": return_embeddings,
        "skip_last_bn": skip_last_bn,
        "normalize_feats": normalize_feats,
        "activation_name": activation_name,
        "use_weight_norm_prototypes": use_weight_norm_prototypes,
    }

    head = MODEL_HEADS_REGISTRY["swav_head"](**cfg)
    return head


def register_vissl_heads(register: FlashRegistry):
    register(swav_head)

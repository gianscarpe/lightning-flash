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
import functools
from typing import Any, Callable, Dict, List, Optional, Sequence, Union

import torch
import torch.nn as nn
from attrdict.dictionary import AttrDict
from torch.utils.data import DataLoader, Sampler

from flash.core.adapter import Adapter
from flash.core.data.auto_dataset import BaseAutoDataset
from flash.core.data.data_source import DefaultDataKeys
from flash.core.model import Task
from flash.core.utilities.imports import _VISSL_AVAILABLE
from flash.core.utilities.url_error import catch_url_error
from flash.image.embedding.heads import vissl_heads

if _VISSL_AVAILABLE:
    from classy_vision.losses import build_loss, ClassyLoss
    from classy_vision.tasks import ClassyTask
    from vissl.models.base_ssl_model import BaseSSLMultiInputOutputModel

    from flash.core.integrations.vissl.hooks import AdaptVISSLHooks


class MockVISSLTask:
    def __init__(self, vissl_loss, task_config) -> None:
        self.loss = vissl_loss
        self.config = task_config


class VISSLAdapter(Adapter, AdaptVISSLHooks):
    """The ``VISSLAdapter`` is an :class:`~flash.core.adapter.Adapter` for integrating with VISSL."""

    required_extras: str = "image"

    def __init__(self, vissl_trunk: nn.Module, vissl_heads: nn.Module, vissl_loss: ClassyLoss, **kwargs):
        # TODO: take in hooks and pass to super so that it can call it
        Adapter.__init__(self)
        AdaptVISSLHooks.__init__(self, **kwargs)

        self.vissl_trunk = vissl_trunk
        self.vissl_heads = [vissl_heads]
        self.vissl_loss = vissl_loss
        # mock config here
        self.vissl_task = MockVISSLTask(vissl_loss, config)

    @classmethod
    @catch_url_error
    def from_task(
        cls,
        task: Task,
        loss_fn: ClassyLoss,
        backbone: nn.Module,
        embedding_dim: int,
        heads: Union[nn.Module, List[nn.Module]],
        **kwargs,
    ) -> Adapter:
        # model_config = AttrDict({})
        # optimizer_config = AttrDict({})
        # vissl_model = BaseSSLMultiInputOutputModel(model_config, optimizer_config)
        # vissl_model.base_model = backbone
        # vissl_model.heads = nn.ModuleList([heads] if not isinstance(heads, List) else heads)
        return cls(vissl_trunk=backbone, vissl_heads=heads, vissl_loss=loss_fn, hooks=kwargs["hooks"])

    def forward(self, batch) -> Any:
        # TODO: ["res5", ["AdaptiveAvgPool2d", [[1, 1]]]] for CNNs
        if isinstance(batch[DefaultDataKeys.INPUT], List):
            out = self.multi_res_input_forward(batch[DefaultDataKeys.INPUT], [], self.vissl_heads)
        else:
            out = self.single_input_forward(batch[DefaultDataKeys.INPUT], [], self.vissl_heads)

        return out

    def single_input_forward(self, batch, feature_names, heads):
        feats = self.vissl_trunk(batch, feature_names)
        return self.heads_forward(feats, heads)

    def multi_res_input_forward(self, batch, feature_names, heads):
        assert isinstance(batch, list)
        idx_crops = torch.cumsum(
            torch.unique_consecutive(torch.tensor([inp.shape[-1] for inp in batch]), return_counts=True)[1],
            0,
        )

        feats = []
        start_idx = 0
        # in order to optimize memory usage, we can do single pass per
        # crop as well. Set the flag to be true.
        # if self.model_config.SINGLE_PASS_EVERY_CROP:
        idx_crops = torch.Tensor(list(range(1, 1 + len(batch)))).int()

        for end_idx in idx_crops:
            feat = self.vissl_trunk(torch.cat(batch[start_idx:end_idx]), feature_names)
            start_idx = end_idx
            assert len(feat) == 1
            feats.append(feat[0])
        feats = [torch.cat(feats)]
        return self.heads_forward(feats, heads)

    def heads_forward(self, feats, heads):
        # Example case: training linear classifiers on various layers
        if len(feats) == len(heads):
            output = []
            for feat, head in zip(feats, heads):
                out = head(feat)
                if isinstance(out, List):
                    output.extend(out)
                else:
                    output.append(out)
            return output
        # Example case: Head consisting of several layers
        elif (len(heads) > 1) and (len(feats) == 1):
            output = feats[0]
            for head in heads:
                output = head(output)
            # our model is multiple output.
            return [output]
        else:
            raise AssertionError(f"Mismatch in #head: {len(heads)} and #features: {len(feats)}")

    def training_step(self, batch: Any, batch_idx: int) -> Any:
        out = self(batch)

        # call forward hook from VISSL (momentum updates)
        for hook in self.hooks:
            hook.on_forward(self.vissl_task)

        exit(-1)

        # out can be torch.Tensor/List target is torch.Tensor
        # loss = self.vissl_loss(out, target)

        # TODO: log
        # TODO: Include call to ClassyHooks during training
        # return loss

    def validation_step(self, batch: Any, batch_idx: int) -> None:
        out = self(batch)

        # out can be torch.Tensor/List target is torch.Tensor
        # loss = self.vissl_loss(out, target)

        # TODO: log
        # TODO: Include call to ClassyHooks during training
        # return loss

    def test_step(self, batch: Any, batch_idx: int) -> None:
        # vissl_input, target = batch
        # out = self(vissl_input)

        # # out can be torch.Tensor/List target is torch.Tensor
        # loss = self.vissl_loss(out, target)

        # # TODO: log
        # # TODO: Include call to ClassyHooks during training
        pass

    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> Any:
        # TODO: return embedding here
        pass

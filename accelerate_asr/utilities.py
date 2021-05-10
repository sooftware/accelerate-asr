# MIT License
#
# Copyright (c) 2021 Soohwan Kim
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import torch
import torch.nn as nn
import platform
from omegaconf import DictConfig, OmegaConf
from torch.optim import Adam, AdamW, SGD, ASGD, Adamax, Adadelta, Adagrad
from torch.optim.lr_scheduler import ReduceLROnPlateau

from accelerate_asr.optim import RAdam, AdamP
from accelerate_asr.optim.lr_scheduler import TriStageLRScheduler, TransformerLRScheduler
from accelerate_asr.optim.lr_scheduler.lr_scheduler import LearningRateScheduler


def check_environment(configs: DictConfig, logger) -> int:
    """
    Check execution envirionment.
    OS, Processor, CUDA version, Pytorch version, ... etc.
    """
    logger.info(f"Operating System : {platform.system()} {platform.release()}")
    logger.info(f"Processor : {platform.processor()}")

    num_devices = torch.cuda.device_count()

    if configs.use_cuda == 'cuda':
        for idx in range(torch.cuda.device_count()):
            logger.info(f"device : {torch.cuda.get_device_name(idx)}")
        logger.info(f"CUDA is available : {torch.cuda.is_available()}")
        logger.info(f"CUDA version : {torch.version.cuda}")
        logger.info(f"PyTorch version : {torch.__version__}")

    else:
        logger.info(f"CUDA is available : {torch.cuda.is_available()}")
        logger.info(f"PyTorch version : {torch.__version__}")

    return num_devices


def get_optimizer(configs: DictConfig, model: nn.Module):
    supported_optimizers = {
        "adam": Adam,
        "adamp": AdamP,
        "radam": RAdam,
        "adagrad": Adagrad,
        "adadelta": Adadelta,
        "adamax": Adamax,
        "adamw": AdamW,
        "sgd": SGD,
        "asgd": ASGD,
    }
    assert configs.optimizer in supported_optimizers.keys(), \
        f"Unsupported Optimizer: {configs.optimizer}\n" \
        f"Supported Optimizers: {supported_optimizers.keys()}"
    return supported_optimizers[configs.optimizer](model.parameters(), lr=configs.lr, weight_decay=1e-5)


def get_lr_scheduler(configs: DictConfig, optimizer) -> LearningRateScheduler:
    if configs.scheduler == 'transformer':
        lr_scheduler = TransformerLRScheduler(
            optimizer,
            peak_lr=configs.peak_lr,
            final_lr=configs.final_lr,
            final_lr_scale=configs.final_lr_scale,
            warmup_steps=configs.warmup_steps,
            decay_steps=configs.decay_steps,
        )
    elif configs.scheduler == 'tri_stage':
        lr_scheduler = TriStageLRScheduler(
            optimizer,
            init_lr=configs.init_lr,
            peak_lr=configs.peak_lr,
            final_lr=configs.final_lr,
            final_lr_scale=configs.final_lr_scale,
            init_lr_scale=configs.init_lr_scale,
            warmup_steps=configs.warmup_steps,
            total_steps=configs.warmup_steps + configs.decay_steps,
        )
    elif configs.scheduler == 'reduce_lr_on_plateau':
        lr_scheduler = ReduceLROnPlateau(
            optimizer,
            patience=configs.lr_patience,
            factor=configs.lr_factor,
        )
    else:
        raise ValueError(f"Unsupported `scheduler`: {configs.scheduler}\n"
                         f"Supported `scheduler`: transformer, tri_stage, reduce_lr_on_plateau")

    return lr_scheduler

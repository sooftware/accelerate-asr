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

import os
import hydra
import logging
import torch.nn as nn
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig
from accelerate import Accelerator

from accelerate_asr.criterion import JointCTCCrossEntropyLoss
from accelerate_asr.data.data_loader import build_data_loader
from accelerate_asr.metric import WordErrorRate, CharacterErrorRate
from accelerate_asr.model.model import ConformerLSTMModel
from accelerate_asr.optim.optimizer import Optimizer
from accelerate_asr.trainer.supervised_trainer import SupervisedTrainer
from accelerate_asr.utilities import (
    get_optimizer,
    get_lr_scheduler,
    check_environment,
)
from accelerate_asr.hydra_configs import (
    DataConfigs,
    SpectrogramConfigs,
    MelSpectrogramConfigs,
    FBankConfigs,
    MFCCConfigs,
    TrainerGPUConfigs,
    TrainerTPUConfigs,
    ReduceLROnPlateauLRSchedulerConfigs,
    TriStageLRSchedulerConfigs,
    TransformerLRSchedulerConfigs,
    ConformerLSTMConfigs,
)


cs = ConfigStore.instance()
cs.store(group="data", name="default", node=DataConfigs)
cs.store(group="audio", name="spectrogram", node=SpectrogramConfigs)
cs.store(group="audio", name="melspectrogram", node=MelSpectrogramConfigs)
cs.store(group="audio", name="fbank", node=FBankConfigs)
cs.store(group="audio", name="mfcc", node=MFCCConfigs)
cs.store(group="model", name="conformer_lstm", node=ConformerLSTMConfigs)
cs.store(group="lr_scheduler", name="reduce_lr_on_plateau", node=ReduceLROnPlateauLRSchedulerConfigs)
cs.store(group="lr_scheduler", name="tri_stage", node=TriStageLRSchedulerConfigs)
cs.store(group="lr_scheduler", name="transformer", node=TransformerLRSchedulerConfigs)
cs.store(group="trainer", name="gpu", node=TrainerGPUConfigs)
cs.store(group="trainer", name="tpu", node=TrainerTPUConfigs)


@hydra.main(config_path=os.path.join('..', "configs"), config_name="train")
def hydra_entry(configs: DictConfig) -> None:
    logger = logging.getLogger(__name__)
    check_environment(configs, logger)
    data_loader, vocab = build_data_loader(configs)

    wer_metric = WordErrorRate(vocab)
    cer_metric = CharacterErrorRate(vocab)

    accelerator = Accelerator(fp16=True if configs.fp16 else False)
    device = accelerator.device

    model = ConformerLSTMModel(configs=configs, vocab=vocab, num_classes=len(vocab))
    model = nn.DataParallel(model).to(device)

    optimizer = get_optimizer(configs, model)
    lr_scheduler = get_lr_scheduler(configs, optimizer)

    optimizer = Optimizer(optim=optimizer,
                          scheduler=lr_scheduler,
                          accelerator=accelerator,
                          scheduler_period=configs.max_epochs * len(data_loader['train']),
                          gradient_clip_val=configs.gradient_clip_val)
    criterion = JointCTCCrossEntropyLoss(num_classes=len(vocab),
                                         ignore_index=vocab.pad_id,
                                         ctc_weight=configs.ctc_weight,
                                         cross_entropy_weight=configs.cross_entropy_weight,
                                         blank_id=vocab.blank_id).to(device)

    trainer = SupervisedTrainer(model=model,
                                optimizer=optimizer,
                                criterion=criterion,
                                data_lodaer=data_loader,
                                accelerator=accelerator,
                                device=device,
                                wer_metric=wer_metric,
                                cer_metric=cer_metric,
                                logger=logger,
                                max_epochs=configs.max_epochs)
    trainer.fit()


if __name__ == '__main__':
    hydra_entry()

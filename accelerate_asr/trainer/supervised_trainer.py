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
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import torch
import torch.nn as nn
from logging import Logger
from torch.optim import Optimizer
from accelerate import Accelerator

from accelerate_asr.metric import WordErrorRate, CharacterErrorRate


class SupervisedTrainer:
    def __init__(
            self,
            model: nn.Module,
            optimizer: Optimizer,
            criterion: nn.Module,
            data_lodaer: dict,
            accelerator: Accelerator,
            device: torch.device,
            wer_metric: WordErrorRate,
            cer_metric: CharacterErrorRate,
            max_epochs: int,
            logger: Logger,
    ) -> None:
        self._model = model
        self._optimizer = optimizer
        self._criterion = criterion
        self._data_loader = data_lodaer
        self._accelerator = accelerator
        self._device = device
        self._wer_metric = wer_metric
        self._cer_metric = cer_metric
        self._max_epochs = max_epochs
        self._logger = logger

    def _train_epoches(self, epoch):
        self._model, self._optimizer, self._data_loader['train'] = self._accelerator.prepare(
            self._model,
            self._optimizer,
            self._data_loader['train'],
        )
        epoch_time_steps = len(self._data_loader['train'])
        self._model.train()

        for idx, (inputs, targets, input_lengths, target_lengths) in enumerate(self._data_loader['train']):
            inputs = inputs.to(self._device)
            targets = targets.to(self._device)
            input_lengths = input_lengths.to(self._device)
            target_lengths = target_lengths.to(self._device)

            self._optimizer.zero_grad()
            y_hats, encoder_log_probs, encoder_output_lengths = self._model(inputs, input_lengths)

            max_target_length = targets.size(1) - 1  # minus the start of sequence symbol
            y_hats = y_hats[:, :max_target_length, :]

            loss, ctc_loss, cross_entropy_loss = self._criterion(
                encoder_log_probs=encoder_log_probs.transpose(0, 1),
                decoder_log_probs=y_hats.contiguous().view(-1, y_hats.size(-1)),
                output_lengths=encoder_output_lengths,
                targets=targets[:, 1:],
                target_lengths=target_lengths,
            )

            wer = self._wer_metric(targets[:, 1:], y_hats)
            cer = self._cer_metric(targets[:, 1:], y_hats)

            self._logger.info(f"Epoch: {epoch} Stage: train Loss: {loss} CTC-Loss: {ctc_loss} "
                              f"CrossEntropy-Loss: {cross_entropy_loss} "
                              f"WER: {wer} CER: {cer} Steps: {idx}/{epoch_time_steps}")

            self._accelerator.backward(loss)
            self._optimizer.step()

    def _validate(self, epoch):
        self._model.eval()

        for split in ['val-clean', 'val-other']:
            self._model, self._optimizer, self._data_loader[split] = self._accelerator.prepare(
                self._model,
                self._optimizer,
                self._data_loader[split],
            )
            total_steps = len(self._data_loader[split])

            for idx, (inputs, targets, input_lengths, target_lengths) in enumerate(self._data_loader[split]):
                inputs = inputs.to(self._device)
                targets = targets.to(self._device)
                input_lengths = input_lengths.to(self._device)
                target_lengths = target_lengths.to(self._device)

                y_hats, encoder_log_probs, encoder_output_lengths = self._model(inputs, input_lengths)

                max_target_length = targets.size(1) - 1  # minus the start of sequence symbol
                y_hats = y_hats[:, :max_target_length, :]

                loss, ctc_loss, cross_entropy_loss = self._criterion(
                    encoder_log_probs=encoder_log_probs.transpose(0, 1),
                    decoder_log_probs=y_hats.contiguous().view(-1, y_hats.size(-1)),
                    output_lengths=encoder_output_lengths,
                    targets=targets[:, 1:],
                    target_lengths=target_lengths,
                )

                wer = self._wer_metric(targets[:, 1:], y_hats)
                cer = self._cer_metric(targets[:, 1:], y_hats)

                self._logger.info(f"Epoch: {epoch} Stage: {split} Loss: {loss} CTC-Loss: {ctc_loss} "
                                  f"CrossEntropy-Loss: {cross_entropy_loss} "
                                  f"WER: {wer} CER: {cer} Steps: {idx}/{total_steps}")

    def _test(self):
        self._model.eval()

        for split in ['test-clean', 'test-other']:
            self._model, self._optimizer, self._data_loader[split] = self._accelerator.prepare(
                self._model,
                self._optimizer,
                self._data_loader[split],
            )
            total_steps = len(self._data_loader[split])

            for idx, (inputs, targets, input_lengths, target_lengths) in enumerate(self._data_loader[split]):
                inputs = inputs.to(self._device)
                targets = targets.to(self._device)
                input_lengths = input_lengths.to(self._device)
                target_lengths = target_lengths.to(self._device)

                y_hats, encoder_log_probs, encoder_output_lengths = self._model(inputs, input_lengths)

                max_target_length = targets.size(1) - 1  # minus the start of sequence symbol
                y_hats = y_hats[:, :max_target_length, :]

                loss, ctc_loss, cross_entropy_loss = self._criterion(
                    encoder_log_probs=encoder_log_probs.transpose(0, 1),
                    decoder_log_probs=y_hats.contiguous().view(-1, y_hats.size(-1)),
                    output_lengths=encoder_output_lengths,
                    targets=targets[:, 1:],
                    target_lengths=target_lengths,
                )

                wer = self._wer_metric(targets[:, 1:], y_hats)
                cer = self._cer_metric(targets[:, 1:], y_hats)

                self._logger.info(f"Stage: {split} Loss: {loss} CTC-Loss: {ctc_loss} "
                                  f"CrossEntropy-Loss: {cross_entropy_loss} "
                                  f"WER: {wer} CER: {cer} Steps: {idx}/{total_steps}")

    def fit(self):
        for epoch in range(self._max_epochs):
            self._train_epoches(epoch)
            self._validate(epoch)
            self._accelerator.save(self._model, f"Epoch-{epoch}.pt")
        self._test()

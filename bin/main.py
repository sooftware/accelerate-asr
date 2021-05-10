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
from omegaconf import DictConfig
from torch import optim
from accelerate import Accelerator

from accelerate_asr.criterion import JointCTCCrossEntropyLoss
from accelerate_asr.data.data_loader import BucketingSampler, AudioDataLoader
from accelerate_asr.data.librispeech.downloader import LibriSpeechDownloader
from accelerate_asr.data.dataset import (
    parse_manifest_file,
    SpectrogramDataset,
    MelSpectrogramDataset,
    MFCCDataset,
    FBankDataset,
)
from accelerate_asr.metric import WordErrorRate, CharacterErrorRate
from accelerate_asr.model.model import ConformerLSTMModel


@hydra.main(config_path=os.path.join('..', "configs"), config_name="train")
def hydra_entry(configs: DictConfig) -> None:
    dataset, data_loader = dict(), dict()
    splits = ['train', 'val-clean', 'val-other', 'test-clean', 'test-other']
    manifest_paths = [
        f"{configs.dataset_path}/train-960.txt",
        f"{configs.dataset_path}/dev-clean.txt",
        f"{configs.dataset_path}/dev-other.txt",
        f"{configs.dataset_path}/test-clean.txt",
        f"{configs.dataset_path}/test-other.txt",
    ]

    if configs.feature_extract_method == 'spectrogram':
        audio_dataset = SpectrogramDataset
    elif configs.feature_extract_method == 'melspectrogram':
        audio_dataset = MelSpectrogramDataset
    elif configs.feature_extract_method == 'mfcc':
        audio_dataset = MFCCDataset
    elif configs.feature_extract_method == 'fbank':
        audio_dataset = FBankDataset
    else:
        raise ValueError(f"Unsupported `feature_extract_method`: {configs.feature_extract_method}")

    logger = logging.getLogger(__name__)
    downloader = LibriSpeechDownloader(dataset_path=configs.dataset_path,
                                       logger=logger,
                                       librispeech_dir=configs.librispeech_dir)
    vocab = downloader.download(configs.vocab_size)

    for idx, (path, split) in enumerate(zip(manifest_paths, splits)):
        audio_paths, transcripts = parse_manifest_file(path)
        dataset[split] = audio_dataset(
            dataset_path=configs.dataset_path,
            audio_paths=audio_paths,
            transcripts=transcripts,
            sos_id=vocab.sos_id,
            eos_id=vocab.eos_id,
            apply_spec_augment=configs.apply_spec_augment if idx == 0 else False,
            sample_rate=configs.sample_rate,
            num_mels=configs.num_mels,
            frame_length=configs.frame_length,
            frame_shift=configs.frame_shift,
            freq_mask_para=configs.freq_mask_para,
            freq_mask_num=configs.freq_mask_num,
            time_mask_num=configs.time_mask_num,
        )
        sampler = BucketingSampler(dataset[split], batch_size=configs.batch_size)
        data_loader[split] = AudioDataLoader(
            dataset=dataset[split],
            num_workers=configs.num_workers,
            batch_sampler=sampler,
            batch_size=configs.batch_size,
            shuffle=True,
        )

    epoch_steps = len(data_loader)

    wer_metric = WordErrorRate(vocab)
    cer_metric = CharacterErrorRate(vocab)

    accelerator = Accelerator()
    device = accelerator.device

    model = ConformerLSTMModel(configs=configs,
                               num_classes=len(vocab),
                               vocab=vocab)
    model = nn.DataParallel(model).to(device)

    optimizer = optim.Adam(model.parameters(), lr=configs.lr, weight_decay=1e-5)
    criterion = JointCTCCrossEntropyLoss(num_classes=len(vocab),
                                         ignore_index=vocab.pad_id,
                                         ctc_weight=configs.ctc_weight,
                                         cross_entropy_weight=configs.cross_entropy_weight,
                                         blank_id=configs.blank_id).to(device)

    for epoch in range(configs.max_epochs):
        model, optimizer, data_loader['train'] = accelerator.prepare(model, optimizer, data_loader['train'])
        model.train()

        for idx, (inputs, targets, input_lengths, target_lengths) in enumerate(data_loader['train']):
            inputs = inputs.to(device)
            targets = targets.to(device)
            input_lengths = input_lengths.to(device)
            target_lengths = target_lengths.to(device)

            optimizer.zero_grad()
            y_hats, encoder_log_probs, encoder_output_lengths = model(inputs, input_lengths)

            max_target_length = targets.size(1) - 1  # minus the start of sequence symbol
            y_hats = y_hats[:, :max_target_length, :]

            loss, ctc_loss, cross_entropy_loss = criterion(
                encoder_log_probs=encoder_log_probs.transpose(0, 1),
                decoder_log_probs=y_hats.contiguous().view(-1, y_hats.size(-1)),
                output_lengths=encoder_output_lengths,
                targets=targets[:, 1:],
                target_lengths=target_lengths,
            )

            wer = wer_metric(targets[:, 1:], y_hats)
            cer = cer_metric(targets[:, 1:], y_hats)

            logger.info(f"Epoch: {epoch} Stage: Train Loss: {loss} CTC-Loss: {ctc_loss} "
                        f"CrossEntropy-Loss: {cross_entropy_loss} "
                        f"WER: {wer} CER: {cer} Steps: {idx}/{epoch_steps}")

            accelerator.backward(loss)
            optimizer.step()

        model.eval()

        for split in splits[1:]:
            model, optimizer, data_loader[split] = accelerator.prepare(model, optimizer, data_loader[split])

            for idx, (inputs, targets, input_lengths, target_lengths) in enumerate(data_loader[split]):
                inputs = inputs.to(device)
                targets = targets.to(device)
                input_lengths = input_lengths.to(device)
                target_lengths = target_lengths.to(device)

                optimizer.zero_grad()
                y_hats, encoder_log_probs, encoder_output_lengths = model(inputs, input_lengths)

                max_target_length = targets.size(1) - 1  # minus the start of sequence symbol
                y_hats = y_hats[:, :max_target_length, :]

                loss, ctc_loss, cross_entropy_loss = criterion(
                    encoder_log_probs=encoder_log_probs.transpose(0, 1),
                    decoder_log_probs=y_hats.contiguous().view(-1, y_hats.size(-1)),
                    output_lengths=encoder_output_lengths,
                    targets=targets[:, 1:],
                    target_lengths=target_lengths,
                )

                wer = wer_metric(targets[:, 1:], y_hats)
                cer = cer_metric(targets[:, 1:], y_hats)

                logger.info(f"Epoch: {epoch} Stage: Train Loss: {loss} CTC-Loss: {ctc_loss} "
                            f"CrossEntropy-Loss: {cross_entropy_loss} "
                            f"WER: {wer} CER: {cer} Steps: {idx}/{epoch_steps}")

                accelerator.backward(loss)
                optimizer.step()


if __name__ == '__main__':
    hydra_entry()

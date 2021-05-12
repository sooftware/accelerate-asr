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
import numpy as np
import logging
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from torch.utils.data.sampler import Sampler
from typing import Tuple

from accelerate_asr.data.librispeech.downloader import LibriSpeechDownloader
from accelerate_asr.data.librispeech.preprocess import generate_manifest_files
from accelerate_asr.vocabs import LibriSpeechVocabulary
from accelerate_asr.vocabs.vocab import Vocabulary
from accelerate_asr.data.dataset import (
    parse_manifest_file,
    SpectrogramDataset,
    MelSpectrogramDataset,
    MFCCDataset,
    FBankDataset,
)


def build_data_loader(configs: DictConfig) -> Tuple[dict, Vocabulary]:
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
    if configs.dataset_download:
        downloader = LibriSpeechDownloader(dataset_path=configs.dataset_path,
                                           logger=logger,
                                           librispeech_dir=configs.librispeech_dir)
        downloader.download(configs.vocab_size)

    generate_manifest_files(configs, configs.vocab_size, logger)
    vocab = LibriSpeechVocabulary("tokenizer.model", configs.vocab_size)

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
        )

    return data_loader, vocab


def _collate_fn(batch, pad_id: int = 0):
    """ functions that pad to the maximum sequence length """
    def seq_length_(p):
        return len(p[0])

    def target_length_(p):
        return len(p[1])

    # sort by sequence length for rnn.pack_padded_sequence()
    batch = sorted(batch, key=lambda sample: sample[0].size(0), reverse=True)

    seq_lengths = [len(s[0]) for s in batch]
    target_lengths = [len(s[1]) - 1 for s in batch]

    max_seq_sample = max(batch, key=seq_length_)[0]
    max_target_sample = max(batch, key=target_length_)[1]

    max_seq_size = max_seq_sample.size(0)
    max_target_size = len(max_target_sample)

    feat_size = max_seq_sample.size(1)
    batch_size = len(batch)

    seqs = torch.zeros(batch_size, max_seq_size, feat_size)

    targets = torch.zeros(batch_size, max_target_size).to(torch.long)
    targets.fill_(pad_id)

    for x in range(batch_size):
        sample = batch[x]
        tensor = sample[0]
        target = sample[1]
        seq_length = tensor.size(0)

        seqs[x].narrow(0, 0, seq_length).copy_(tensor)
        targets[x].narrow(0, 0, len(target)).copy_(torch.LongTensor(target))

    seq_lengths = torch.IntTensor(seq_lengths)
    target_lengths = torch.IntTensor(target_lengths)

    return seqs, targets, seq_lengths, target_lengths


class AudioDataLoader(DataLoader):
    """ Audio Data Loader """
    def __init__(
            self,
            dataset: torch.utils.data.Dataset,
            num_workers: int,
            batch_sampler: torch.utils.data.sampler.Sampler,
            **kwargs,
    ) -> None:
        super(AudioDataLoader, self).__init__(
            dataset=dataset,
            num_workers=num_workers,
            batch_sampler=batch_sampler,
            **kwargs,
        )
        self.collate_fn = _collate_fn


class BucketingSampler(Sampler):
    """ Samples batches assuming they are in order of size to batch similarly sized samples together. """
    def __init__(self, data_source, batch_size: int = 32, drop_last: bool = False) -> None:
        super(BucketingSampler, self).__init__(data_source)
        self.batch_size = batch_size
        self.data_source = data_source
        ids = list(range(0, len(data_source)))
        self.bins = [ids[i:i + batch_size] for i in range(0, len(ids), batch_size)]
        self.drop_last = drop_last

    def __iter__(self):
        for ids in self.bins:
            np.random.shuffle(ids)
            yield ids

    def __len__(self):
        return len(self.bins)

    def shuffle(self, epoch):
        np.random.shuffle(self.bins)

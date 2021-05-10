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
import wget
import tarfile
import shutil
from logging import Logger

from accelerate_asr.vocabs import LibriSpeechVocabulary
from accelerate_asr.vocabs.vocab import Vocabulary
from accelerate_asr.data.librispeech.preprocess import (
    collect_transcripts,
    prepare_tokenizer,
    generate_manifest_file,
)


class LibriSpeechDownloader:
    librispeech_parts = [
        'dev-clean',
        'test-clean',
        'dev-other',
        'test-other',
        'train-clean-100',
        'train-clean-360',
        'train-other-500',
    ]

    def __init__(
            self,
            dataset_path: str,
            librispeech_dir: str,
            logger: Logger,
    ):
        self.dataset_path = dataset_path
        self.librispeech_dir = librispeech_dir
        self.logger = logger

    def _download_librispeech(self) -> None:
        """
        Download librispeech dataset.
            - train-960(train-clean-100, train-clean-360, train-other-500)
            - dev-clean
            - dev-other
            - test-clean
            - test-other
        """
        base_url = "http://www.openslr.org/resources/12"
        train_dir = "train-960"

        if not os.path.exists(self.dataset_path):
            os.mkdir(self.dataset_path)

        for part in self.librispeech_parts:
            self.logger.info(f"Librispeech-{part} download..")
            url = f"{base_url}/{part}.tar.gz"
            wget.download(url, self.dataset_path)

            self.logger.info(f"Un-tarring archive {self.dataset_path}/{part}.tar.gz")
            tar = tarfile.open(f"{self.dataset_path}/{part}.tar.gz", mode="r:gz")
            tar.extractall()
            tar.close()
            os.remove(f"{self.dataset_path}/{part}.tar.gz")

        self.logger.info("Merge all train packs into one")

        if not os.path.exists(os.path.join(self.dataset_path, self.librispeech_dir)):
            os.mkdir(os.path.join(self.dataset_path, self.librispeech_dir))
        if not os.path.exists(os.path.join(self.dataset_path, self.librispeech_dir, train_dir)):
            os.mkdir(os.path.join(self.dataset_path, self.librispeech_dir, train_dir))

        for part in self.librispeech_parts[:-3]:  # dev, test
            shutil.move(
                os.path.join(self.librispeech_dir, part),
                os.path.join(self.dataset_path, self.librispeech_dir, part),
            )

        for part in self.librispeech_parts[-3:]:  # train
            path = os.path.join(self.librispeech_dir, part)
            subfolders = os.listdir(path)
            for subfolder in subfolders:
                shutil.move(
                    os.path.join(path, subfolder),
                    os.path.join(self.dataset_path, self.librispeech_dir, train_dir, subfolder),
                )

    def _generate_manifest_files(self, vocab_size: int) -> None:
        """
        Generate manifest files.
        Format: {audio_path}\t{transcript}\t{numerical_label}

        Args:
            vocab_size (int): size of subword vocab

        Returns:
            None
        """
        self.logger.info("Generate Manifest Files..")
        transcripts_collection = collect_transcripts(
            os.path.join(self.dataset_path, self.librispeech_dir),
            self.librispeech_dir,
        )
        prepare_tokenizer(transcripts_collection[0], vocab_size)

        for idx, part in enumerate(['train-960', 'dev-clean', 'dev-other', 'test-clean', 'test-other']):
            generate_manifest_file(self.dataset_path, part, transcripts_collection[idx])

    def download(self, vocab_size: int = 5000) -> Vocabulary:
        self._download_librispeech()
        self._generate_manifest_files(vocab_size)
        return LibriSpeechVocabulary("tokenizer.model", vocab_size)

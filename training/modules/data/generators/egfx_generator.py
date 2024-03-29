import os
import numpy as np
import soundfile as sf
import torch
from torch.utils.data import TensorDataset

import config
from training.modules.utils.file_system import get_source_target_file_paths, convert_audio_files_to_arrays
from training.modules.utils.array_utils import zeropad_tuple_to_longest_item
from training.modules.data.dataset_generator import DatasetGenerator


class EGFxDatasetGenerator(DatasetGenerator):
    def __init__(self,
                 input_audio_folders: list,
                 output_audio_folders: list,
                 samplerate=44100,
                 block_size=1,
                 normalize_amp=False):
        super().__init__()
        assert len(input_audio_folders) == len(output_audio_folders), \
            'number of input_audio_folders does not match number of output_audio_folders'

        self.input_audio_folders = input_audio_folders
        self.output_audio_folders = output_audio_folders
        self.samplerate = samplerate
        self.block_size = block_size
        self.normalize_amp = normalize_amp

    def generate_dataset(self, normalize_amp=None, samplerate=None, block_size=None):
        """
        Generates the complete dataset from which
        training, validation and testing data will be retrieved
        returns a TensorDataset object suitable for use with Torches dataloade
        """

        self.normalize_amp = normalize_amp or self.normalize_amp
        self.samplerate = samplerate or self.samplerate
        self.block_size = block_size or self.block_size

        input_arrays = []
        target_arrays = []
        for (input_folder, output_folder) in zip(self.input_audio_folders, self.output_audio_folders):
            assert os.path.exists(input_folder), "Input audio folder not found " + self.input_audio_folders
            assert os.path.exists(output_folder), "Output audio folder not found " + self.output_audio_folders
            print('\ninput_folder:\n', input_folder)
            print('output_folder:\n', output_folder)

            # get audio files in the input folder
            input_file_paths, output_file_paths = get_source_target_file_paths(
                input_folder,
                output_folder,
                ".wav"
            )

            assert len(input_file_paths) > 0, "get_source_target_file_paths yielded zero inputs files"
            assert len(output_file_paths) > 0, "get_source_target_file_paths yielded zero outputs files"
            _, sample_rate = sf.read(input_file_paths[0], dtype='float32')
            if sample_rate != config.SAMPLE_RATE:
                print("load_wav_file warning: sample rate wrong, resampling from ", sample_rate, "to", config.SAMPLE_RATE)

            # get wav files as np arrays
            input_arrays += convert_audio_files_to_arrays(
                audio_file_paths=input_file_paths,
                samplerate=self.samplerate
            )
            target_arrays += convert_audio_files_to_arrays(
                audio_file_paths=output_file_paths,
                samplerate=self.samplerate
            )

        # zeropad everything to match longest - obtain arrays of uniform length
        input_arrays, target_arrays = zeropad_tuple_to_longest_item((input_arrays, target_arrays))

        assert len(input_arrays) == len(target_arrays)
        assert all(len(input_arrays[0]) == len(arr) for arr in input_arrays[1:])
        assert all(len(target_arrays[0]) == len(arr) for arr in target_arrays[1:])
        assert all(len(input_arrays[0]) == len(arr) for arr in target_arrays[0:])

        # optional normalization
        if self.normalize_amp:
            input_max = max(np.max(input_arrays), np.abs(np.min(input_arrays)))
            target_max = max(np.max(target_arrays), np.abs(np.min(target_arrays)))
            combine_max = max(input_max, target_max)
            # Normalize audio between -1.0 and +1.0
            input_arrays /= combine_max
            target_arrays /= combine_max

        print("generate_dataset: Loaded frames from audio file", len(input_file_paths))
        # normal shape for LSTM input: (sequence_length, batch_size, input_size]
        input_tensor = torch.tensor(np.array(input_arrays))
        output_tensor = torch.tensor(np.array(target_arrays))

        input_tensor = torch.unsqueeze(input_tensor, dim=-1)
        output_tensor = torch.unsqueeze(output_tensor, dim=-1)

        input_tensor, output_tensor = self._reshape_block_size(input_tensor, output_tensor)
        print("input dataset tensors' shape", input_tensor.shape)

        dataset = TensorDataset(input_tensor, output_tensor)
        self.dataset = dataset
        return dataset

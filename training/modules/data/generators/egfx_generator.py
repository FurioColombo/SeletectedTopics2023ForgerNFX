import numpy as np
import os
from torch.utils.data import TensorDataset
import torch

from modules.utils.file_system import get_source_target_file_paths, convert_audio_files_to_arrays
from modules.utils.array_utils import zeropad_tuple_to_longest_item
from modules.data.dataset_generator import DatasetGenerator


class EGFxDatasetGenerator(DatasetGenerator):

    def __init__(self,
                 input_audio_folder,
                 output_audio_folder,
                 samplerate=44100,
                 normalize_amp=False):
        super().__init__()
        self.input_audio_folder = input_audio_folder
        self.output_audio_folder = output_audio_folder
        self.samplerate = samplerate
        self.normalize_amp = normalize_amp

    def generate_dataset(self):  # , pre_emphasis_filter=True):
        """
        Generates the complete dataset from which
        training, validation and testing data will be retrieved
        Wright used half second segments for each data point
        returns a  TensorDataset object suitable for use with Torches dataloader:
        dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
        """
        assert os.path.exists(self.input_audio_folder), "Input audio folder not found " + self.input_audio_folder
        assert os.path.exists(self.output_audio_folder), "Output audio folder not found " + self.output_audio_folder
        # get audio files in the input folder
        input_file_paths, output_file_paths = get_source_target_file_paths(self.input_audio_folder, self.output_audio_folder,
                                                                           ".wav")

        print('input_file_paths:\n', input_file_paths)
        print('out_file_paths:\n', output_file_paths)
        assert len(input_file_paths) > 0, "get_files_in_folder yielded zero inputs files"
        assert len(output_file_paths) > 0, "get_files_in_folder yielded zero outputs files"

        # get wav files as np arrays
        input_arrays = convert_audio_files_to_arrays(
            audio_file_paths=input_file_paths,
            samplerate=self.samplerate
        )
        target_arrays = convert_audio_files_to_arrays(
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

        print("generate_dataset:: Loaded frames from audio file", len(input_file_paths))
        # Convert input and output fragments to PyTorch tensors
        # nothing that the normal shape for an input to an LSTM
        # is (sequence_length, batch_size, input_size]
        # so input_fragments[0]

        input_tensor = torch.tensor(np.array(input_arrays))
        output_tensor = torch.tensor(np.array(target_arrays))

        input_tensor = torch.unsqueeze(input_tensor, dim=-1)
        output_tensor = torch.unsqueeze(output_tensor, dim=-1)

        assert input_tensor.size() == output_tensor.size()

        dataset = TensorDataset(input_tensor, output_tensor)
        self.dataset = dataset
        return dataset
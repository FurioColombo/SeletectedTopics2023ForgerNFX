import os
import numpy as np
import torch
from torch.utils.data import TensorDataset
from modules.utils.file_system import load_wav_file, get_file_paths_in_folder
from modules.data.dataset_generator import DatasetGenerator


# based on MYK dataset generation
class FragmentsDatasetGenerator(DatasetGenerator):

    def __init__(self,
                 input_audio_folder,
                 output_audio_folder,
                 samplerate=44100,
                 frag_len_seconds=2):
        super().__init__()
        self.input_audio_folder = input_audio_folder
        self.output_audio_folder = output_audio_folder
        self.samplerate = samplerate
        self.frag_len_seconds = frag_len_seconds

    def _audio_file_to_fragments(self, audio_filepath):
        """
        load in the sent audio file and chop it into fragments of the sent length
        """
        assert os.path.exists(audio_filepath), "Cannot find audio file " + audio_filepath
        # audio_data, sr = sf.read(audio_filepath)

        audio_data = load_wav_file(audio_filepath, self.samplerate)
        num_samples_per_frag = int(self.frag_len_seconds * self.samplerate)
        num_frags = int(np.ceil(len(audio_data) / num_samples_per_frag))

        fragments = []
        for i in range(num_frags):
            frag_start = i * num_samples_per_frag
            frag_end = (i + 1) * num_samples_per_frag
            fragment = audio_data[frag_start:frag_end]
            if len(fragment) != num_samples_per_frag:
                continue
            fragments.append(fragment)

        return fragments

    def _audio_filelist_to_fragments(self, audio_files):
        """
        iterates the sent list of audio files
        loads their data, chops to fragments
        returns a list of all the fragments
        """
        all_fragments = []
        for file in audio_files:
            fragments = self._audio_file_to_fragments(file)
            all_fragments.extend(fragments)
        return all_fragments

    def generate_dataset(self):
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
        input_files = get_file_paths_in_folder(self.input_audio_folder, ".wav")
        output_files = get_file_paths_in_folder(self.output_audio_folder, ".wav")
        assert len(input_files) > 0, "get_files_in_folder yielded zero inputs files"
        assert len(output_files) > 0, "get_files_in_folder yielded zero outputs files"

        input_fragments = self._audio_filelist_to_fragments(input_files)
        output_fragments = self._audio_filelist_to_fragments(output_files)

        assert len(input_fragments) > 0, "get_files_in_folder yielded zero inputs"
        assert len(output_fragments) > 0, "get_files_in_folder yielded zero outputs"

        # make lengths the same
        if len(input_fragments) > len(output_fragments):
            input_fragments = input_fragments[0:len(output_fragments)]
        else:
            output_fragments = output_fragments[0:len(input_fragments)]
        print("generate_dataset:: Loaded frames from audio file", len(input_fragments))
        # Convert input and output fragments to PyTorch tensors
        # noting that the normal shape for an input to an LSTM
        # is (sequence_length, batch_size, input_size]
        # so input_fragments[0]
        input_tensor = torch.tensor(np.array(input_fragments))
        print("input tensor shape", input_tensor.shape)
        output_tensor = torch.tensor(np.array(output_fragments))
        dataset = TensorDataset(input_tensor, output_tensor)
        self.dataset = dataset
        return dataset

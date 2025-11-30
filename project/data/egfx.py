import os
import torch
import numpy as np
import torchaudio
from torch.utils.data import Dataset
from typing import List, Tuple, Optional

class EGFxDataset(Dataset):
    def __init__(
        self, 
        input_root: str, 
        output_root: str, 
        block_size: int = 512, 
        sample_rate: int = 44100,
        normalize: bool = True
    ):
        self.block_size = block_size
        self.sample_rate = sample_rate
        self.normalize = normalize
        
        self.input_files, self.output_files = self._find_paired_files(input_root, output_root)
        
        if len(self.input_files) == 0:
            raise ValueError(f"No paired files found in {input_root} and {output_root}")
            
        print(f"Found {len(self.input_files)} paired files.")
        
        self.input_data, self.output_data = self._load_and_process_data()

    def _find_paired_files(self, input_dir: str, output_dir: str) -> Tuple[List[str], List[str]]:
        input_files = []
        output_files = []
        
        # Walk through input directory
        for root, _, files in os.walk(input_dir):
            for file in files:
                if file.endswith('.wav'):
                    # Get relative path from input_dir
                    rel_path = os.path.relpath(os.path.join(root, file), input_dir)
                    
                    # Check if corresponding file exists in output_dir
                    out_path = os.path.join(output_dir, rel_path)
                    
                    if os.path.exists(out_path):
                        input_files.append(os.path.join(input_dir, rel_path))
                        output_files.append(out_path)
        
        # Sort to ensure deterministic order
        # Zip, sort, unzip
        if not input_files:
            return [], []
            
        combined = sorted(zip(input_files, output_files))
        input_files, output_files = zip(*combined)
        
        return list(input_files), list(output_files)

    def _load_audio(self, path: str) -> torch.Tensor:
        waveform, sr = torchaudio.load(path)
        
        # Resample if needed
        if sr != self.sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
            waveform = resampler(waveform)
            
        # Mono
        if waveform.shape[0] > 1:
            waveform = waveform[0:1, :]
            
        return waveform

    def _load_and_process_data(self) -> Tuple[torch.Tensor, torch.Tensor]:
        input_list = []
        output_list = []
        
        max_len = 0
        
        # Load all files
        for in_path, out_path in zip(self.input_files, self.output_files):
            in_wave = self._load_audio(in_path)
            out_wave = self._load_audio(out_path)
            
            # Trim leading zeros (simple heuristic from original code)
            # Original code used np.trim_zeros which trims 0s from front and back
            # Let's stick to simple loading for now, maybe trim later if needed
            # But original code did trim zeros. Let's do it to match behavior if important.
            # However, torch tensors don't have trim_zeros. Convert to numpy?
            # Let's keep it simple and see if it works. 
            
            len_val = min(in_wave.shape[1], out_wave.shape[1])
            in_wave = in_wave[:, :len_val]
            out_wave = out_wave[:, :len_val]
            
            input_list.append(in_wave)
            output_list.append(out_wave)
            max_len = max(max_len, len_val)
            
        # Pad to max length
        final_input_list = []
        final_output_list = []
        
        for in_wave, out_wave in zip(input_list, output_list):
            pad_len = max_len - in_wave.shape[1]
            if pad_len > 0:
                in_wave = torch.nn.functional.pad(in_wave, (0, pad_len))
                out_wave = torch.nn.functional.pad(out_wave, (0, pad_len))
            final_input_list.append(in_wave)
            final_output_list.append(out_wave)
            
        # Concatenate all files
        # Shape: (1, total_samples)
        full_input = torch.cat(final_input_list, dim=1)
        full_output = torch.cat(final_output_list, dim=1)
        
        # Normalize
        if self.normalize:
            max_val = max(full_input.abs().max(), full_output.abs().max())
            if max_val > 0:
                full_input = full_input / max_val
                full_output = full_output / max_val
                
        # Chunk into blocks
        # (1, total_samples) -> (num_blocks, 1, block_size)
        num_blocks = full_input.shape[1] // self.block_size
        
        full_input = full_input[:, :num_blocks * self.block_size]
        full_output = full_output[:, :num_blocks * self.block_size]
        
        # Reshape: (num_blocks, block_size, 1) -> transpose to (num_blocks, 1, block_size)
        # Original code: (sequence_length, batch_size, input_size) for LSTM?
        # My LSTM expects (batch, channels, time) or (batch, time, channels)
        # Let's stick to (batch, channels, time) which is (num_blocks, 1, block_size)
        
        input_blocks = full_input.view(1, num_blocks, self.block_size).permute(1, 0, 2)
        output_blocks = full_output.view(1, num_blocks, self.block_size).permute(1, 0, 2)
        
        return input_blocks, output_blocks

    def __len__(self):
        return self.input_data.shape[0]

    def __getitem__(self, idx):
        return self.input_data[idx], self.output_data[idx]

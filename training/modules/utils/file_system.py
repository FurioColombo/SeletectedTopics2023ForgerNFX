import soundfile as sf
import numpy as np
import librosa
import os


def get_all_file_paths_in_folder(folder, extension='.wav'):
    """
    returns a list of files in the sent folder with the sent extension
    """
    file_list = []
    for file in os.listdir(folder):
        if file.endswith(extension):
            file_list.append(os.path.join(folder, file))
    return file_list


def get_all_file_names_in_folder(folder, extension='.wav'):
    """
    returns a list of files in the sent folder with the sent extension
    """
    file_list = []
    for file in os.listdir(folder):
        if file.endswith(extension):
            file_list.append(file)
    return file_list


def get_source_target_file_paths(sources_folder, targets_folder, extension='.wav'):
    """
    returns a list of files in the sent folder with the sent extension
    """
    source_file_names_list = set(get_all_file_names_in_folder(sources_folder))
    source_file_path_list = []

    target_file_names_list = set(get_all_file_names_in_folder(targets_folder))
    target_file_path_list = []

    common_file_names = source_file_names_list.intersection(target_file_names_list)
    for file_name in common_file_names:
        source_file_path_list.append(os.path.join(sources_folder, file_name))
        target_file_path_list.append(os.path.join(targets_folder, file_name))

    print("Discarded", len(source_file_names_list) - len(common_file_names),
          'source files due to missing target correspondence')
    print("Discarded", len(target_file_names_list) - len(common_file_names),
          'target files due to missing source correspondence')

    return source_file_path_list, target_file_path_list


# ======================== AUDIO 2 NPARRAY ========================
def load_wav_file(filename, target_samplerate):
    """
    Load a WAV file using the soundfile module, resample to 44100 Hz, and return the first channel.
    """
    # Load the WAV file
    data, samplerate = sf.read(filename, dtype='float32')

    # Resample to 44100 Hz
    if samplerate != target_samplerate:
        print("load_wav_file warning: sample rate wrong, resampling from ", samplerate, "to", target_samplerate)
        data = librosa.resample(
            data,
            orig_sr=samplerate,
            target_sr=target_samplerate,
            res_type='soxr_hq'  # todo: use best one?
        )

    # If the file has more than one channel, only return the first channel
    if len(data.shape) > 1 and data.shape[1] > 1:
        data = data[:, 0]

    # Put each sample in its own array
    # so we have [[sample1], [sample2]]
    data = np.array(data)
    data = data[:, np.newaxis]

    return data


def convert_audio_files_to_arrays(audio_file_paths, samplerate, zeropad=False):
    audio_arrays = []
    for i in range(len(audio_file_paths)):
        # array = _audio_file2padded_array(audio_file_paths[i], samplerate) #todo: add zeropad to longest sequence
        array = load_wav_file(audio_file_paths[i], samplerate)
        array = np.ravel(array)
        audio_arrays.append(array)
    return audio_arrays

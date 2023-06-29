import soundfile as sf
import librosa
import numpy as np


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


def zeropad2longest(np_arrays_tuple: tuple):
    assert len(np_arrays_tuple) != 0
    print('zeropad2longest')
    if type(np_arrays_tuple) is not tuple:
        np_arrays_tuple = (np_arrays_tuple, )

    max_len = 0
    for np_arrays in np_arrays_tuple:
        assert len(np_arrays) != 0
        for idx, np_array in enumerate(np_arrays):
            # delete eventual zeros at the beginning of the audio file
            np_arrays[idx] = np.trim_zeros(np_array)

        max_len = max(map(len, np_arrays))

    for np_arrays in np_arrays_tuple:
        for idx, np_array in enumerate(np_arrays):
            # zero-pad input to get desired common sample length
            np_arrays[idx] = np.pad(
                np_array,
                (0, max_len - len(np_array)),
                mode='constant',
                constant_values=0
            )
    # todo: move this
    for np_arrays in np_arrays_tuple:
        for idx, np_array in enumerate(np_arrays):
            # zero-pad input to get desired common sample length
            np_arrays[idx] = np_array[0:220500]

    return np_arrays_tuple

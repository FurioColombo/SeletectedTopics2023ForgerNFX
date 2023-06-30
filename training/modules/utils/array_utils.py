import numpy as np


def zeropad_tuple_to_longest_item(np_arrays_tuple: tuple, max_length=None):
    assert len(np_arrays_tuple) != 0
    if type(np_arrays_tuple) is not tuple:
        np_arrays_tuple = (np_arrays_tuple, )

    max_len = 0
    # ensure input has correct data structure, save max length
    for np_arrays in np_arrays_tuple:
        assert len(np_arrays) != 0
        for idx, np_array in enumerate(np_arrays):
            # delete eventual zeros at the beginning of the audio file
            np_arrays[idx] = np.trim_zeros(np_array)

        max_len = max(map(len, np_arrays))

    # zero-padding
    for np_arrays in np_arrays_tuple:
        for idx, np_array in enumerate(np_arrays):
            # zero-pad input to get desired common sample length
            np_arrays[idx] = np.pad(
                np_array,
                (0, max_len - len(np_array)),
                mode='constant',
                constant_values=0
            )
            if max_length is not None:
                np_arrays[idx] = np_array[0:max_length]

    return np_arrays_tuple

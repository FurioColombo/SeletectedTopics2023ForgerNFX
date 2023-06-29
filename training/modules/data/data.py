from torch.utils.data import TensorDataset
from torch.utils.data import random_split
import numpy as np


def get_train_valid_test_datasets(dataset, splits=None):
    if splits is None:
        splits = [0.8, 0.1, 0.1]
    assert type(dataset) == TensorDataset, "dataset should be a TensorDataset but it is " + str(type(dataset))
    assert np.sum(splits) == 1, "Splits do not add up to one"
    assert (len(dataset) * np.min(splits)) > 1, "Smallest split yields zero size " + str(splits) + " over " + str(
        len(dataset))
    train_size = int(splits[0] * len(dataset))
    val_size = int(splits[1] * len(dataset))
    test_size = int(splits[2] * len(dataset))

    assert train_size > 0, "Trying to create training data but got zero length"
    assert val_size > 0, "Trying to create validation data but got zero length"
    assert test_size > 0, "Trying to create test data but got zero length"
    # now only use as much of the dataset as we need, in case splits are 
    req_items = np.sum([train_size, val_size, test_size])
    if req_items > len(dataset):
        diff = len(dataset) - req_items
        train_size -= diff  # hit the big one
        print("Cannot get that many items from the dataset: want", req_items, "got", len(dataset),
              " trimming the big one by ", diff)

    if req_items < len(dataset):
        diff = req_items - len(dataset)
        train_size -= diff  # hit the big one
        print("Cannot get that many items from the dataset: want", req_items, "got", len(dataset),
              " trimming the big one by ", diff)

    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])
    return train_dataset, val_dataset, test_dataset

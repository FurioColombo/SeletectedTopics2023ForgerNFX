from abc import ABC, abstractmethod
from torch.utils.data import TensorDataset
from torch.utils.data import random_split
import numpy as np


class DatasetGenerator(ABC):

    def __init__(self):
        self.dataset = None

    @abstractmethod
    def generate_dataset(self):
        pass

    def get_train_valid_test_datasets(self, splits=None):
        if splits is None:
            splits = [0.8, 0.1, 0.1]
        assert type(self.dataset) == TensorDataset, "dataset should be a TensorDataset but it is " + str(type(self.dataset))
        assert np.sum(splits) == 1, "Splits do not add up to one"
        assert (len(self.dataset) * np.min(splits)) > 1, "Smallest split yields zero size " + str(splits) + " over " + str(
            len(self.dataset))
        assert self.dataset is not None

        train_size = int(splits[0] * len(self.dataset))
        val_size = int(splits[1] * len(self.dataset))
        test_size = int(splits[2] * len(self.dataset))

        assert train_size > 0, "Trying to create training data but got zero length"
        assert val_size > 0, "Trying to create validation data but got zero length"
        assert test_size > 0, "Trying to create test data but got zero length"
        # now only use as much of the dataset as we need, in case splits are
        req_items = np.sum([train_size, val_size, test_size])
        if req_items > len(self.dataset):
            diff = len(self.dataset) - req_items
            train_size -= diff  # hit the big one
            print("Cannot get that many items from the dataset: want", req_items, "got", len(self.dataset),
                  " trimming the big one by ", diff)

        if req_items < len(self.dataset):
            diff = req_items - len(self.dataset)
            train_size -= diff  # hit the big one
            print("Cannot get that many items from the dataset: want", req_items, "got", len(self.dataset),
                  " trimming the big one by ", diff)

        train_dataset, val_dataset, test_dataset = random_split(self.dataset, [train_size, val_size, test_size])
        return train_dataset, val_dataset, test_dataset




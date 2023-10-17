from math import ceil
from secrets import SystemRandom
from torch.utils.data import Sampler
from typing import List, Iterator
import torch


class GroupSampler(Sampler):

    def __init__(self, data_length: int, batch_size: int, group_size: int, randomizer: SystemRandom = SystemRandom()) -> None:
        self.data_length = data_length
        self.group_size = group_size
        self.batch_size = batch_size
        self.randomizer = randomizer

    def __len__(self) -> int:
        return ceil((self.data_length + self.batch_size * self.group_size - 1) / (self.batch_size * self.group_size))

    def group_batches(self) -> int:
        return (self.data_length + self.group_size - 1) // self.group_size

    def batch(self, iterable):
        l = len(iterable)
        n = self.batch_size
        for ndx in range(0, l, n):
            yield iterable[ndx:min(ndx + n, l)]

    def __iter__(self) -> Iterator[List[int]]:
        sizes = torch.tensor(range(1, self.data_length+1))
        groups = list(torch.chunk(torch.argsort(sizes), self.group_batches()))
        self.randomizer.shuffle(groups)

        l = len(groups)
        n = self.batch_size

        for ndx in range(0, l, n):
            batch = []
            for tensor in groups[ndx:min(ndx + n, l)]:
                indices = tensor.tolist()
                batch.extend(indices)
            yield batch
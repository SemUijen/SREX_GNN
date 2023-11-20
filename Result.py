from torch import Tensor
from typing import List, Tuple, Optional
from plotting.plot_srex_parameters import plot_srex_parameters


class Result:

    def __init__(self, epoch):
        self.epoch = epoch
        self.binary_label: List[Tensor] = []
        self.output: List[Tensor] = []
        self.config_shape: List[Tuple[int, int, int]] = []
        self.instances: List[int] = []

    def add(self, label: Tensor, output: Tensor, shape: Tuple[int, int, int], instance_id: Tensor):
        self.binary_label.append(label)
        self.output.append(output.detach())
        self.config_shape.append(shape)
        self.instances.append(instance_id.item())

    def plot(self, idx: int, only_pos: Optional[bool] = True, lim_labels: Optional[bool] = False):
        label = self.binary_label[idx]
        output = self.output[idx]
        shape = self.config_shape[idx]
        instance_name = self.instance_name(self.instances[idx])
        if lim_labels:
            indices = self.get_lim_indices(idx)
            plot_srex_parameters(label, indices=indices, title=f"label: {instance_name}", lim_labels=True)
            plot_srex_parameters(output, indices=indices, title=f"label: {instance_name}", lim_labels=True)

        else:
            plot_srex_parameters(label.view(shape).numpy(), title=f"label: {instance_name}")
            plot_srex_parameters(output.view(shape).numpy(), title=f"output: {instance_name}")

    def get_lim_indices(self, idx):
        max_move, p1, p2 = self.config_shape[idx]
        list_test = []
        for i in range(max_move):
            for i2 in range(p1):
                if i2 < p2:
                    list_test.append([i, i2, i2])
                else:
                    list_test.append([i, i2, 0])

        return list_test

    @staticmethod
    def instance_name(key: int):
        instance_dict = {0: 'X-n439-k37', 1:'X-n393-k38',
                         2: 'X-n449-k29', 3: 'ORTEC-n405-k18',
                         4: 'ORTEC-n510-k23', 5: 'X-n573-k30',
                         6: 'ORTEC-VRPTW-ASYM-0bdff870-d1-n458-k35',
                         7: 'R2_8_9',  8: 'R1_4_10'}

        return instance_dict[key]
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

    def plot(self, idx: int, only_pos: Optional[bool] = True):
        label = self.binary_label[idx]
        output = self.output[idx]
        shape = self.config_shape[idx]
        instance_name = self.instance_name(self.instances[idx])
        plot_srex_parameters(label.view(shape).numpy(), title=f"label: {instance_name}", only_pos=only_pos)
        plot_srex_parameters(output.view(shape).numpy(), title=f"output: {instance_name}", only_pos=only_pos)


    @staticmethod
    def instance_name(key: int):
        instance_dict = {0: 'X-n439-k37', 1:'X-n393-k38',
                         2: 'X-n449-k29', 3: 'ORTEC-n405-k18',
                         4: 'ORTEC-n510-k23', 5: 'X-n573-k30',
                         6: 'ORTEC-VRPTW-ASYM-0bdff870-d1-n458-k35',
                         7: 'R2_8_9',  8: 'R1_4_10'}

        return instance_dict[key]
r"""
K-Median wrapper
"""

import pathlib
import numpy as np
from typing import Union, List

from ml4co_kit.task.base import TASK_TYPE
from ml4co_kit.wrapper.base import WrapperBase
from ml4co_kit.task.facility.kmedian import KMedian
from ml4co_kit.utils.time_utils import tqdm_by_time
from ml4co_kit.utils.file_utils import check_file_path

class KMedianWrapper(WrapperBase):
    def __init__(self, precision: Union[np.float32, np.float64] = np.float32):
        super().__init__(KMedianWrapper, self).__init__(
            task_type=TASK_TYPE.KMEDIAN, precision=precision
        )
        self.task_list: List[KMedian] = list()

    def from_txt(
            self,
            file_path: pathlib.Path,
            ref: bool = False,
            overwrite: bool = True,
            normalize: bool = False,
            show_time: bool = False
    ):
        """Read task data from ''.txt.'' file"""
        if overwrite:
            self.task_list = list()

        with open(file_path, "r") as file:
            load_msg = f"Loading data from {file_path}"
            for idx, line in tqdm_by_time(enumerate(file), load_msg, show_time):
                line = line.strip()

                split_0 = line.split("k ")[1]
                split_1 = split_0.split(" client_locs ")
                k = int(split_1[0])

                split_2 = split_1[1].split(" facilities locs ")
                client_locs = split_2[0]

                split_3 = split_2[1].split(" demands ")
                facilities_locs = split_3[0]

                split_4 = split_3[1].split(" output ")
                demands = split_4[0]
                sol = split_4[1] if len(split_4) > 1 else None

                # Parse client_locs
                client_locs = client_locs.split(" ")
                client_locs = np.array(
                    [
                        [float(client_locs[i]), float(client_locs[i+1])]
                        for i in range(0, len(client_locs), 2)
                    ],
                    dtype=self.precision
                )

                # Parse facilites_locs
                facilities_locs = facilities_locs.split(" ")
                facilities_locs = np.array(
                    [
                        [facilities_locs[i], facilities_locs[i+1]]
                        for i in range(0, len(facilities_locs), 2)
                    ],
                    dtype=self.precision
                )

                # Parse demands
                demands = demands.split(" ")
                demands = np.array(
                    [float(d) for d in demands], dtype=self.precision
                )

                # Parse solution
                if sol is not None:
                    sol = sol.split(" ")
                    sol = np.array([int(s) for s in sol], dtype=int)

                if overwrite:
                    task = KMedian(precision=self.precision)
                else:
                    task = self.task_list[idx]

                task.from_data(
                    k=k,
                    client_locs=client_locs,
                    facilities_locs=facilities_locs,
                    demands=demands,
                    sol=sol,
                    ref=ref,
                    normalize=normalize
                )

                if overwrite:
                    self.task_list.append(task)

    def to_txt(self, file_path: pathlib.Path, show_time:bool = False, mode: str = "w"):
        """Write task data to ''.txt'' file"""
        # Check file path
        check_file_path(file_path)

        # Save task data to ''.txt'' file
        with open(file_path, mode) as f:
            write_msg = f"Writing data to {file_path}"
            for task in tqdm_by_time(self.task_list, write_msg, show_time):
                task._check_sol_not_none()

                k = task.k
                client_locs = task.client_locs
                facilities_locs = task.facilities_locs
                demands = task.demands
                sol = task.sol

                f.write("k " + str(k))
                f.write(" client_locs ")
                f.write(" ".join(str(x) + " " + str(y) for x, y in client_locs))
                f.write(" facilities_locs ")
                f.write(" ".join(str(x) + " " + str(y) for x, y in facilities_locs))
                f.write(" demands ")
                f.write(" ".join(str(d) for d in demands))
                f.write(" output ")
                f.write(" ".join(str(s) for s in sol))
                f.write("\n")
            f.close()
""""
Generator for kmedian FLP instances
"""

import numpy as np
from enum import Enum
from typing import Union

from ml4co_kit.task.base import TASK_TYPE
from ml4co_kit.generator.base import GeneratorBase
from ml4co_kit.task.facility.kmedian import KMedian

class KMEDIAN_TYPE(str, Enum):
    """Define the k-median instance types"""
    UNIFORM = "uniform"
    GAUSIIAN = "gaussian"

class KMedianGenerator(GeneratorBase):
    """Generator for k-median facility location instances"""

    def __init__(
            self,
            distribution_type: KMEDIAN_TYPE = KMEDIAN_TYPE.UNIFORM,
            precision: Union[np.float32, np.float64] = np.float32,
            # problem sizes
            client_num: int = 100,
            facilities_num: int = 30,
            k: int = 5,
            # demads
            min_demand: int = 1,
            max_demand: int = 9,
            # gaussian
            gaussian_mean_x: float = 0.0,
            gaussian_mean_y: float = 0.0,
            gaussian_std: float = 1.0,
    ):
        super(KMedianGenerator, self).__init__(
            task_type=TASK_TYPE.KMEDIAN,
            distribution_type=distribution_type,
            precision=precision
        )

        # Initialize Attributes
        self.client_num = client_num
        self.facilities_num = facilities_num
        self.k = k

        # Demands
        self.min_demand = min_demand
        self.max_demand = max_demand

        # Gaussian params
        self.gaussian_mean_x = gaussian_mean_x
        self.gaussian_mean_y = gaussian_mean_y
        self.gaussian_std = gaussian_std

        # Generation function dictionary
        self.generate_func_dict = {
            KMEDIAN_TYPE.UNIFORM: self._generate_uniform,
            KMEDIAN_TYPE.GAUSIIAN: self._generate_gaussian,
        }

    def _generate_demands(self) -> np.ndarray:
        """Generate demands for each client"""
        return np.random.randint(
            self.min_demand, self.max_demand, (self.client_num, )
        )
    
    def _generate_uniform(self) -> KMedian:
        # Generate coordinates uniformly in [0, 1]
        client_locs = np.random.uniform(0.0, 1.0, (self.client_num, 2))
        facilities_locs = np.random.uniform(0.0, 1.0, (self.facilities_num, 2))
        demands = self._generate_demands()

        # Create isntance
        task_data = KMedian(precision=self.precision)
        task_data.from_data(
            k=self.k,
            client_locs=client_locs,
            facilities_locs=facilities_locs,
            demands=demands
        )
        return task_data
    
    def _generate_gaussian(self) -> KMedian:
        # Generate coordinates according to a gaussian distribution
        client_locs = np.random.normal((self.gaussian_mean_x, self.gaussian_mean_y), self.gaussian_std, (self.client_num, 2))
        facilities_locs = np.random.normal((self.gaussian_mean_x, self.gaussian_mean_y), self.gaussian_std, (self.facilities_num, 2))
        demands = self._generate_demands()

        # Create instance
        task_data = KMedian(precision=self.precision)
        task_data.from_data(
            k=self.k,
            client_locs=client_locs,
            facilities_locs=facilities_locs,
            demands=demands
        )
        return task_data
    

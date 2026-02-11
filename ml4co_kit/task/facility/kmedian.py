r"""
The k-median version of the facility location optimization problem

The k-median FLOP deals with selecting k facilities from a list of potential locations in order
to minimize the total demand-weighted distance from each client to its nearest facility
"""

import pathlib
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Union
from ml4co_kit.task.base import TaskBase, TASK_TYPE
from ml4co_kit.utils.file_utils import check_file_path

class KMedian(TaskBase):
    def __init__(
            self,
            k: Optional[int] = None,
            precision: Union[np.float32, np.float64] = np.float32
    ):
        super().__init__(task_type=TASK_TYPE.KMEDIAN, minimize=True, precision=precision)

        # Initialize Attributes
        self.k = k
        self.client_num: Optional[int] = None
        self.facilities_num: Optional[int] = None

        self.client_locs: Optional[np.ndarray] = None               # (clients_num, coord_dim)
        self.facilities_locs: Optional[np.ndarray] = None           # (facilities_num, coord_dim)
        self.dists: Optional[np.ndarray] = None                     # (clients_num, facilites_num)
        self.demands: Optional[np.ndarray] = None                   # (clients_num, )

    def _check_k_not_none(self):
        """Check if k has been specified"""
        if self.k is None:
            raise ValueError("''k'' cannot be None!")
        
    def _check_dists_dim(self):
        """Check the correct dimensionality of the dist array"""
        if self.dists.ndim != 2:
            raise ValueError("''dists'' should be a 2D array with shape (clients_num, facilites_num)")

    def _check_demands_dim(self):
        """Check the correct dimensionality of demands array"""
        if self.demands.ndim != 1:
            raise ValueError("''demands'' should be a 1D array with shape (clients_num, )")
        
    def _check_client_locs_dim(self):
        """Check correct dimensionality of client locs (should be 2D)"""
        if self.client_locs.ndim != 2:
            raise ValueError("''client_locs'' should be a 2D array with shape (clients_num, coord_dim)")
        
    def _check_facilities_locs_dim(self):
        """Check correct dimensionality of facilities locs (should be 2D)"""
        if self.facilities_locs.ndim != 2:
            raise ValueError("''facilities_locs'' should be a 2D array with shape (facilites_num, coord_dim)")
        
    def _normalize_dists(self):
        """Normalize the distances into a [0,1] range"""
        min_value = np.min(self.dists)
        max_value = np.max(self.dists)
        if max_value > min_value:
            self.dists = (self.dists - min_value) / (max_value - min_value)

    def _compute_dists_from_coords(self):
        if self.client_locs == None or self.facilities_locs == None:
            raise ValueError("Both client and facilities locs need to be defined")
        self._check_client_locs_dim()
        self._check_facilities_locs_dim()
        # Use euclidean distance
        diff = self.client_locs[:, None, :] - self.facilities_locs[None, :, :]
        self.dists = np.linalg.norm(diff, axis=-1).astype(self.precision)

    def _get_open_indices(self, sol: np.ndarray) -> np.ndarray:
        """Get the indices of the facilities to be open in a given solution"""
        if sol.ndim != 1:
            raise ValueError("Solution should be a 1D array")
        
        # Two supported formats:
        # 1) Indices list of length k
        # 2) Binary selection vector of length m
        if self.facilities_num == None:
            raise ValueError("''facilities_num'' is not set")

        if sol.size == self.k:
            open_idx = sol.astype(int)
        elif sol.size == self.facilities_num:
            open_idx = np.where(sol.astype(np.bool_))[0]
        else:
            raise ValueError("solution is not the correct format")

        return open_idx
    
    def from_data(
            self,
            *,
            k: Optional[int] = None,
            dists: Optional[np.ndarray] = None,
            client_locs: Optional[np.ndarray] = None, 
            facilities_locs: Optional[np.ndarray] = None,
            demands: Optional[np.ndarray] = None,
            sol: Optional[np.ndarray] = None,
            ref: bool = False,
            normalize: bool = False,
            name: Optional[str] = None
    ):
        # Set attributes and check correct dimensions
        if k is not None:
            self.k = int(k)
        if dists is not None:
            self.dists = dists.astype(self.precision)
            self._check_dists_dim()
        if client_locs is not None:
            self.client_locs = client_locs.astype(self.precision)
            self._check_client_locs_dim()
        if facilities_locs is not None:
            self.facilities_locs = facilities_locs.astype(self.precision)
            self._check_facilities_locs_dim()
        if demands is not None:
            self.demands = demands.astype(self.precision)
            self._check_demands_dim()
        
        # Collect the number of facility locations and client locations
        if self.dists is not None:
            self.client_num, self.facilities_num = self.dists.shape
        elif self.client_locs is not None and self.facilities_locs is not None:
            self.client_num = self.client_locs.shape[0]
            self.facilities_num = self.facilities_locs.shape[0]
    
        # Compute the distance matrix
        if self.dists is None and self.client_locs is not None and self.facilities_locs is not None:
            self._compute_dists_from_coords()

        # Set equal demands if demands was not submitted
        if self.demands is None and self.client_num is not None:
            self.demands = np.ones(self.client_num, dtype=self.precision)
        
        # Check demands is correct shape
        if self.demands is not None and self.client_num is not None:
            if self.demands.shape[0] != self.client_num:
                raise ValueError("The demands vector should have the same size as the number of clients")
            
        if normalize and self.dists is not None:
            self._normalize_dists()

        if sol is not None:
            if ref:
                self.ref_sol = sol.astype(int)
            else:
                self.sol = sol.astype(int)

        if name is not None:
            self.name = name

    def check_constraints(self, sol: np.ndarray) -> bool:
        """Check if the solution is valid"""
        # k facilities were opened
        self._check_k_not_none()
        if self.facilities_num is None:
            raise ValueError("''facilites_num'' is not set")
        
        if sol.ndim != 1:
            return False
        
        # Solution is in binary
        if sol.size == self.facilities_num:
            if not np.all(np.logical_or(sol == 0, sol == 1)):
                return False
            if int(np.sum(sol)) != self.k:
                return False
            return True

        # Solution is indices
        if sol.size == self.k:
            if np.any(sol < 0) or np.any(sol > self.facilities_num):
                return False
            if len(np.unqiue(sol)) != self.k:
                return False
            return True
        
        # Solution is in incompatible format
        return False

    def evaluate(self, sol: np.ndarray) -> np.floating:
        """Evaluate the given kmedian solution"""
        if not self.check_constraints(sol):
            raise ValueError("Invalid solution!")
        
        if self.dists is None:
            raise ValueError("''dists'' is not set")
        
        open_idx = self._get_open_indices(sol)
        # Get the distance to the nearest facility for each client
        min_d = np.min(self.dists[:, open_idx], axis=1)
        return np.sum(min_d * self.demands).astype(self.precision)
    
    def render(
            self,
            save_path: pathlib.Path,
            with_sol: bool = True,
            figsize: tuple = (5.5),
            client_color: str = "tab:blue",
            facility_color: str = "tab:gray",
            open_facility_color: str = "tab:red",
            client_size: int = 20,
            facility_size: int = 40,
    ):
        """Render the KMedian instance with or without solution"""
        check_file_path(save_path)

        if self.client_locs is None or self.facilities_locs is None:
            raise ValueError("''client_locs'' and ''facilities_locs'' are required to render")
        
        sol = self.sol
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111)

        ax.scatter(
            self.client_locs[:, 0], self.client_locs[:, 1],
            c=client_color, s=client_size, label="Clients"
        )

        if with_sol:
            if sol is None:
                raise ValueError("Solution is not provided")
            open_idx = self._get_open_indices(sol)
            closed_idx = np.setdiff1d(np.arange(self.facilities_num), open_idx)

            if closed_idx.size > 0:
                ax.scatter(
                    self.facilities_locs[closed_idx, 0], self.facilities_locs[closed_idx, 1],
                    c=facility_color, s=facility_size, labe="Closed facilites"
                )

            ax.scatter(
                self.facilities_locs[open_idx, 0], self.facilities_locs[open_idx, 1],
                c=open_facility_color, s=facility_size, label="Open facilites"
            )

        else:
            ax.scatter(
                self.facilities_locs[:, 0], self.facilities_locs[:, 1],
                c=facility_color, s=facility_size, label="Facilities"
            )

        ax.legend()
        ax.set_title("K-Median")
        plt.savefig(save_path)
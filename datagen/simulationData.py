import csv
import os

import numpy as np
import torch
from torch.utils.data import Dataset


class SimulationDatasetFromFile(Dataset):
    """
    A generic class for simulation data loading and extraction, as well as pre-filtering of interventions
    NOTE: the 0-th regime should always be the observational one
    """

    def __init__(
        self,
        file_path,
        i_dataset,
        intervention=True,
        fraction_regimes_to_ignore=None,
        regimes_to_ignore=None,
        load_ignored=False,
    ) -> None:
        """
        :param str file_path: Path to the data and the DAG
        :param int i_dataset: Exemplar to use (usually in [1,10])
        :param boolean intervention: If True, use interventional data with interventional targets
        :param list regimes_to_ignore: Regimes that are ignored during training
        """
        super(SimulationDatasetFromFile, self).__init__()
        self.file_path = file_path
        self.i_dataset = i_dataset
        self.intervention = intervention
        # load data
        all_data, all_masks, all_regimes = self.load_data()
        # index of all regimes, even if not used in the regimes_to_ignore case
        self.all_regimes_list = np.unique(all_regimes)

        if fraction_regimes_to_ignore is not None or regimes_to_ignore is not None:
            if fraction_regimes_to_ignore is not None and regimes_to_ignore is not None:
                raise ValueError("either fraction or list, not both")
            if fraction_regimes_to_ignore is not None:
                # select fraction to ignore
                np.random.seed(0)
                sampling_list = self.all_regimes_list
                self.regimes_to_ignore = np.random.choice(
                    sampling_list,
                    int(fraction_regimes_to_ignore * len(sampling_list)),
                )
            else:
                self.regimes_to_ignore = regimes_to_ignore

            to_keep = np.array(
                [
                    regime not in self.regimes_to_ignore
                    for regime in np.array(all_regimes)
                ]
            )
            if not load_ignored:
                data = all_data[to_keep]
                masks = [mask for i, mask in enumerate(all_masks) if to_keep[i]]
                regimes = np.array(
                    [regime for i, regime in enumerate(all_regimes) if to_keep[i]]
                )
            else:
                data = all_data[~to_keep]
                masks = [mask for i, mask in enumerate(all_masks) if ~to_keep[i]]
                regimes = np.array(
                    [regime for i, regime in enumerate(all_regimes) if ~to_keep[i]]
                )
        else:
            data = all_data
            masks = all_masks
            regimes = all_regimes

        self.data = data
        self.regimes = regimes
        self.masks = np.array(masks, dtype=object)

        self.num_regimes = np.unique(self.regimes).shape[0]
        self.num_samples = self.data.shape[0]
        self.dim = self.data.shape[1]

    def __getitem__(self, idx):
        if self.intervention:
            # binarize mask from list
            masks_list = self.masks[idx]
            masks = np.ones((self.dim,))
            for j in masks_list:
                masks[j] = 0
            return (
                self.data[idx].astype(np.float32),
                masks.astype(np.float32),
                self.regimes[idx],
            )
        else:
            # put full ones mask
            return (
                self.data[idx].astype(np.float32),
                np.ones_like(self.regimes[idx]).astype(np.float32),
                self.regimes[idx],
            )

    def __len__(self):
        return self.data.shape[0]

    def load_data(self):
        """
        Load the mask, regimes, and data
        """
        if self.intervention:
            name_data = f"data_interv{self.i_dataset}.npy"
        else:
            name_data = f"data{self.i_dataset}.npy"

        # Load data
        self.data_path = os.path.join(self.file_path, name_data)
        data = np.load(self.data_path)

        # Load intervention masks and regimes
        masks = []
        if self.intervention:
            name_data = f"data_interv{self.i_dataset}.npy"
            interv_path = os.path.join(
                self.file_path, f"intervention{self.i_dataset}.csv"
            )
            regimes = np.genfromtxt(
                os.path.join(self.file_path, f"regime{self.i_dataset}.csv"),
                delimiter=",",
            )
            regimes = regimes.astype(int)

            # read masks
            with open(interv_path, "r") as f:
                interventions_csv = csv.reader(f)
                for row in interventions_csv:
                    mask = [int(x) for x in row]
                    masks.append(mask)
        else:
            regimes = np.array([0] * data.shape[0])

        return data, masks, regimes

    def convert_masks(self, idxs):
        """
        Convert mask index to mask vectors
        :param np.ndarray idxs: indices of mask to convert
        :return: masks
        Example:
            if self.masks[i] = [1,4]
                self.dim = 10 then
            masks[i] = [1,0,1,1,0,1,1,1,1,1]
        """
        masks_list = [self.masks[i] for i in idxs]

        masks = torch.ones((idxs.shape[0], self.dim))
        for i, m in enumerate(masks_list):
            for j in m:
                masks[i, j] = 0

        return masks
    
class SimulationDataset(Dataset):
    def __init__(self, datasets, intervention_sets):
        super(SimulationDataset, self).__init__()
        self.datasets = datasets
        self.intervention_sets = intervention_sets 

        self.load_data()
    
    def load_data(self):
        self.data = np.vstack(self.datasets)
        masks_list = list()
        regimes_list = list()
        for i, dataset in enumerate(self.datasets):
            targets = self.intervention_sets[i]
            mask = np.ones_like(dataset)
            regimes_list += [i] * len(dataset)
            if targets[0] != None:
                mask[:, targets] = 0 # 0 - intervened nodes, 1 - purely observed nodes

            masks_list.append(mask)
        
        self.masks = np.vstack(masks_list)
        self.regimes = regimes_list
    
    def __getitem__(self, idx):
        return (
            self.data[idx].astype(np.float32),
            self.masks[idx].astype(np.float32),
            self.regimes[idx]
        )

    def __len__(self):
        return len(self.data)

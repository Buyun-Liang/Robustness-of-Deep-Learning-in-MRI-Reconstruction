from argparse import ArgumentParser
from pathlib import Path
from typing import Callable, Optional, Union
import FastMriDataModule
import fastmri
import torch
import numpy as np
from matplotlib import pyplot as plt
from fastmri.data import CombinedSliceDataset, SliceDataset

def worker_init_fn(worker_id):
    """Handle random seeding for all mask_func."""
    worker_info = torch.utils.data.get_worker_info()
    data: Union[
        SliceDataset, CombinedSliceDataset
    ] = worker_info.dataset  # pylint: disable=no-member
    # for NumPy random seed we need it to be in this range
    seed = worker_info.seed % (2 ** 32 - 1)  # pylint: disable=no-member

    if isinstance(data, CombinedSliceDataset):
        for i, dataset in enumerate(data.datasets):
            if dataset.transform.mask_func is not None:
                dataset.transform.mask_func.rng.seed((seed + i) % (2 ** 32 - 1))
    elif data.transform.mask_func is not None:
        data.transform.mask_func.rng.seed(seed)

class FastMriDataModule():
    """
    Data module class for fastMRI data sets.
    This class handles configurations for training on fastMRI data. It is set
    up to process configurations independently of training modules.
    Note that subsampling mask and transform configurations are expected to be
    done by the main client training scripts and passed into this data module.
    For training with ddp be sure to set distributed_sampler=True to make sure
    that volumes are dispatched to the same GPU for the validation loop.
    """

    def __init__(
        self,
        data_path: Path,
        challenge: str,
        train_transform: Callable,
        val_transform: Callable,
        test_transform: Callable,
        combine_train_val: bool = False,
        test_split: str = "test",
        test_path: Optional[Path] = None,
        sample_rate: float = 1.0,
        use_dataset_cache_file: bool = True,
        batch_size: int = 1,
        num_workers: int = 4,
        distributed_sampler: bool = False,
    ):
        """
        Args:
            data_path: Path to root data directory. For example, if knee/path
                is the root directory with subdirectories multicoil_train and
                multicoil_val, you would input knee/path for data_path.
            challenge: Name of challenge from ('multicoil', 'singlecoil').
            train_transform: A transform object for the training split.
            val_transform: A transform object for the validation split.
            test_transform: A transform object for the test split.
            combine_train_val: Whether to combine train and val splits into one
                large train dataset. Use this for leaderboard submission.
            test_split: Name of test split from ("test", "challenge").
            test_path: An optional test path. Passing this overwrites data_path
                and test_split.
            sample_rate: Fraction of of the training data split to use. Can be
                set to less than 1.0 for rapid prototyping.
            use_dataset_cache_file: Whether to cache dataset metadata. This is
                very useful for large datasets like the brain data.
            batch_size: Batch size.
            num_workers: Number of workers for PyTorch dataloader.
            distributed_sampler: Whether to use a distributed sampler. This
                should be set to True if training with ddp.
        """
        super().__init__()

        self.data_path = data_path
        self.challenge = challenge
        self.train_transform = train_transform
        self.val_transform = val_transform
        self.test_transform = test_transform
        self.combine_train_val = combine_train_val
        self.test_split = test_split
        self.test_path = test_path
        self.sample_rate = sample_rate
        self.use_dataset_cache_file = use_dataset_cache_file
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.distributed_sampler = distributed_sampler

    def _create_data_loader(
        self,
        data_transform: Callable,
        data_partition: str,
        sample_rate: Optional[float] = None,
    ) -> torch.utils.data.DataLoader:
        if data_partition == "train":
            is_train = True
            sample_rate = self.sample_rate if sample_rate is None else sample_rate
        else:
            is_train = False
            sample_rate = 1.0

        # if desired, combine train and val together for the train split
        dataset: Union[SliceDataset, CombinedSliceDataset]
        if is_train and self.combine_train_val:
            data_paths = [
                self.data_path + f"/{self.challenge}_train",
                self.data_path + f"/{self.challenge}_val",
            ]
            data_transforms = [data_transform, data_transform]
            challenges = [self.challenge, self.challenge]
            sample_rates = [sample_rate, sample_rate]
            dataset = CombinedSliceDataset(
                roots=data_paths,
                transforms=data_transforms,
                challenges=challenges,
                sample_rates=sample_rates,
                use_dataset_cache=self.use_dataset_cache_file,
            )
        else:
            if data_partition in ("test", "challenge") and self.test_path is not None:
                data_path = self.test_path
            else:
                data_path = self.data_path + f"/{self.challenge}_{data_partition}"

            dataset = SliceDataset(
                root=data_path,
                transform=data_transform,
                sample_rate=sample_rate,
                challenge=self.challenge,
                use_dataset_cache=self.use_dataset_cache_file,
            )
        # ensure that entire volumes go to the same GPU in the ddp setting
        sampler = None
        if self.distributed_sampler:
            if is_train:
                sampler = torch.utils.data.DistributedSampler(dataset)
            else:
                sampler = fastmri.data.VolumeSampler(dataset)

        dataloader = torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            worker_init_fn=worker_init_fn,
            sampler=sampler,
        )
        return dataloader

    def prepare_data(self):
        # call dataset for each split one time to make sure the cache is set up on the
        # rank 0 ddp process. if not using cache, don't do this
        if self.use_dataset_cache_file:
            if self.test_path is not None:
                test_path = self.test_path
            else:
                test_path = self.data_path + f"/{self.challenge}_test"
            data_paths = [
                self.data_path + f"/{self.challenge}_train",
                self.data_path + f"/{self.challenge}_val",
                test_path,
            ]
            data_transforms = [
                self.train_transform,
                self.val_transform,
                self.test_transform,
            ]
            for i, (data_path, data_transform) in enumerate(
                zip(data_paths, data_transforms)
            ):
                sample_rate = self.sample_rate if i == 0 else 1.0
                _ = SliceDataset(
                    root=data_path,
                    transform=data_transform,
                    sample_rate=sample_rate,
                    challenge=self.challenge,
                    use_dataset_cache=self.use_dataset_cache_file,
                )

    def train_dataloader(self):
        return self._create_data_loader(self.train_transform, data_partition="train")

    def val_dataloader(self):
        return self._create_data_loader(
            self.val_transform, data_partition="val", sample_rate=1.0
        )

    def test_dataloader(self):
        return self._create_data_loader(
            self.test_transform,
            data_partition=self.test_split,
            sample_rate=1.0,
        )
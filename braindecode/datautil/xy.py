import numpy as np
import pandas as pd
import logging
import mne

from braindecode.datasets.base import BaseDataset, BaseConcatDataset
from braindecode.datautil.windowers import (
    create_fixed_length_windows,)

log = logging.getLogger(__name__)


def create_trials_from_X_y(X, y, sfreq, ch_names=None):
    """Create a BaseConcatDataset of BaseDatasets from X and y to be used for
    decoding with skorch and braindecode, where X is a list of pre-cut trials
    and y are corresponding targets.

    Parameters
    ----------
    X: array-like
        list of pre-cut trials as n_trials x n_channels x n_times
    y: array-like
        targets corresponding to the trials
    sfreq: common sampling frequency of all trials
    ch_names: array-like
        channel names of the trials

    Returns
    -------
    base_datasets: BaseConcatDataset
        X and y transformed to a dataset format that is compatible with skorch
        and braindecode
    """
    base_datasets = []
    if sfreq is None:
        log.error("No sampling frequency given.")
    if ch_names is None:
        ch_names = [str(i) for i in range(X.shape[1])]
        log.info(f"No channel names given, set to 0-{X.shape[1]}).")

    for x, target in zip(X, y):
        info = mne.create_info(ch_names=ch_names, sfreq=sfreq)
        raw = mne.io.RawArray(x, info)
        base_dataset = BaseDataset(raw, pd.Series({"target": target}),
                                   target_name="target")
        base_datasets.append(base_dataset)
    base_datasets = BaseConcatDataset(base_datasets)
    return base_datasets


def create_windows_from_X_y(X, y, drop_last_window, sfreq=None, ch_names=None,
        window_size_samples=None, window_stride_samples=None):
    """Create a BaseConcatDataset of WindowsDatasets from X and y to be used for
    decoding with skorch and braindecode, where X is a list of pre-cut trials
    and y are corresponding targets.

    Parameters
    ----------
    X: array-like
        list of pre-cut trials as n_trials x n_channels x n_times
    y: array-like
        targets corresponding to the trials
    sfreq: common sampling frequency of all trials
    ch_names: array-like
        channel names of the trials
    drop_last_window: bool
        whether or not have a last overlapping window, when
        windows/windows do not equally divide the continuous signal
    window_size_samples: int
        window size
    window_stride_samples: int
        stride between windows

    Returns
    -------
    windows_datasets: BaseConcatDataset
        X and y transformed to a dataset format that is compatible with skorch
        and braindecode
    """
    base_datasets = create_trials_from_X_y(
        X=X, y=y, sfreq=sfreq, ch_names=ch_names)
    n_samples_per_x = [x.shape[1] for x in X]
    if window_size_samples is None and window_stride_samples is None:
        if not len(np.unique(n_samples_per_x)) == 1:
            raise ValueError(f"if 'window_size_samples' and "
                             f"'window_stride_samples' are None, "
                             f"all trials have to have the same length")
        window_size_samples = n_samples_per_x[0]
        window_stride_samples = n_samples_per_x[0]
    windows_datasets = create_fixed_length_windows(
        base_datasets,
        start_offset_samples=0,
        stop_offset_samples=0,
        window_size_samples=window_size_samples,
        window_stride_samples=window_stride_samples,
        drop_last_window=drop_last_window
    )
    return windows_datasets

import os
import re
import glob

import numpy as np
import pandas as pd
import mne

from .base import BaseDataset, BaseConcatDataset


class _TUHBase(BaseConcatDataset):
    def __init__(self, path, recording_ids=None, target_name=None,
                 preload=False, add_physician_reports=False):
        all_file_paths = read_all_file_names(
            path, extension='.edf', key=self._time_key)
        if recording_ids is None:
            recording_ids = np.arange(len(all_file_paths))
        self._file_paths = [all_file_paths[rec_id] for rec_id in recording_ids]

        all_base_ds = []
        for i, (recording_id, file_path) in enumerate(zip(
                recording_ids, self._file_paths)):
            raw = mne.io.read_raw_edf(file_path, preload=preload)
            path_splits = file_path.split("/")
            age, gender = _parse_age_and_gender_from_edf_header(file_path)
            # see https://www.isip.piconepress.com/projects/tuh_eeg/downloads/tuh_eeg_abnormal/v2.0.0/_AAREADME.txt
            subject_id = path_splits[-3]
            session_id = path_splits[-1].split("_")[-2][1:]
            segment_id = path_splits[-1].split("_")[-1][1:].split(".")[0]
            d = {'age': age, 'gender': gender, 'subject': subject_id,
                 'recording_id': recording_id, 'session': session_id,
                  'segment': segment_id}
            if add_physician_reports:
                report_paths = read_all_file_names(
                    os.path.dirname(file_path), '.txt', key=self._time_key)
                # there can be several eeg recordings, but there should only be
                # one report per directory
                assert len(report_paths) == 1
                report_path = report_paths[0]
                with open(report_path, "r", encoding="latin-1") as f:
                    physician_report = f.read()
                d["physician_report"] = physician_report
            description = pd.Series(d, name=i)
            base_ds = BaseDataset(raw, description, target_name=target_name)
            all_base_ds.append(base_ds)
        super().__init__(all_base_ds)

    @staticmethod
    def _time_key(file_path):
        # the splits are specific to tuh abnormal eeg data set
        splits = file_path.split('/')
        p = r'(\d{4}_\d{2}_\d{2})'
        [date] = re.findall(p, splits[-2])
        date_id = [int(token) for token in date.split('_')]
        recording_id = _natural_key(splits[-1])
        session_id = re.findall(r'(s\d*)_', (splits[-2]))
        return date_id + session_id + recording_id


class TUH(_TUHBase):
    """Temple University Hospital (TUH) EEG Corpus.

    Parameters
    ----------
    path: str
        parent directory of the dataset
    recording_ids: list(int) | int
        (list of) int of recording(s) to be read (order matters and will
        overwrite default chronological order, e.g. if recording_ids=[1,0],
        then the first recording returned by this class will be chronologically
        later then the second recording. provide recording_ids in ascending
        order to preserve chronological order)
    target_name: str
        will be parsed from the description of the data. could be "age" or
        "gender"
    preload: bool
        if True, preload the data of the Raw objects.
    add_physician_reports: bool
        if True, the physician reports will be read from disk and added to the
        description
    """
    def __init__(self, path, recording_ids=None, target_name=None,
                 preload=False, add_physician_reports=False):
        super().__init__(
            path=path, recording_ids=recording_ids, target_name=target_name,
            preload=preload, add_physician_reports=add_physician_reports)


class TUHAbnormal(_TUHBase):
    """Temple University Hospital (TUH) Abnormal EEG Corpus.

    Parameters
    ----------
    path: str
        parent directory of the dataset
    recording_ids: list(int) | int
        (list of) int of recording(s) to be read (order matters and will
        overwrite default chronological order, e.g. if recording_ids=[1,0],
        then the first recording returned by this class will be chronologically
        later then the second recording. provide recording_ids in ascending
        order to preserve chronological order)
    target_name: str
        can be 'pathological', 'gender', or 'age'
    preload: bool
        if True, preload the data of the Raw objects.
    add_physician_reports: bool
        if True, the physician reports will be read from disk and added to the
        description
    """
    def __init__(self, path, recording_ids=None, target_name="pathological",
                 preload=False, add_physician_reports=False):
        super().__init__(
            path=path, recording_ids=recording_ids, preload=preload,
            add_physician_reports=add_physician_reports)

        for i, file_path in enumerate(self._file_paths):
            path_splits = file_path.split("/")
            if "abnormal" in path_splits:
                pathological = True
            else:
                assert "normal" in path_splits
                pathological = False
            if "train" in path_splits:
                train_or_eval = "train"
            else:
                assert "eval" in path_splits
                train_or_eval = "eval"
            d = {'pathological': pathological, 'train_or_eval': train_or_eval}
            self.datasets[i].description = \
                self.datasets[i].description.append(pd.Series(d))
            self.datasets[i].target_name = target_name
            self.datasets[i].target = self.datasets[i].description[target_name]
        self.description = pd.DataFrame([ds.description for ds in self.datasets])


# TODO: this is very slow. how to improve?
def read_all_file_names(directory, extension, key):
    """Read all files with specified extension from given path and sorts them
    based on a given sorting key.

    Parameters
    ----------
    directory: str
        file path on HDD
    extension: str
        file path extension, i.e. '.edf' or '.txt'
    key: calable
        sorting key for the file paths

    Returns
    -------
    file_paths: list(str)
        a list to all files found in (sub)directories of path
    """
    assert extension.startswith(".")
    file_paths = glob.glob(directory + '**/*' + extension, recursive=True)
    file_paths = sorted(file_paths, key=key)
    assert len(file_paths) > 0, (
        f"something went wrong. Found no {extension} files in {directory}")
    return file_paths


def _natural_key(string):
    pattern = r'(\d+)'
    key = [int(split) if split.isdigit() else None
           for split in re.split(pattern, string)]
    return key


def _parse_age_and_gender_from_edf_header(file_path, return_raw_header=False):
    assert os.path.exists(file_path), f"file not found {file_path}"
    f = open(file_path, 'rb')
    content = f.read(88)
    f.close()
    if return_raw_header:
        return content
    patient_id = content[8:88].decode('ascii')
    [age] = re.findall(r"Age:(\d+)", patient_id)
    [gender] = re.findall(r"\s(\w)\s", patient_id)
    return int(age), gender

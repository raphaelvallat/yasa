"""
Access to YASA tutorial data

The YASA sample dataset is stored on Zenodo
and can be accessed using the Pooch library.
"""

from pathlib import Path

import pooch

__all__ = [
    "fetch_sample",
]


REGISTRY = {
    "sample": {
        "v1": {
            "base_url": "https://zenodo.org/records/14564285/files",
            "registry": {
                "ECG_8hrs_200Hz.npz": "md5:3a05f39925a009e04f0c0c88da4f429e",
                "EOGs_REM_256Hz.npz": "md5:ef86dbef2d6ad99d54aad275175de8e1",
                "N2_spindles_15sec_200Hz.txt": "md5:6601dad681120d3a9ff8ce4f9a9042a0",
                "N3_no-spindles_30sec_100Hz.txt": "md5:9d2ac76f8b2a886da95ffd6a94458861",
                "full_6hrs_100Hz_9channels.npz": "md5:b5faf1b8e1664f4a6f87f12b1144eb7a",
                "full_6hrs_100Hz_Cz+Fz+Pz.npz": "md5:9e12ab265dbc4498989bef0d6acd3e72",
                "full_6hrs_100Hz_hypno.npz": "md5:300c2341bf7625404906d7dda8764add",
                "full_6hrs_100Hz_hypno_30s.txt": "md5:a711266690a7177443b2e724d5d1a495",
                "night_young.edf": "md5:cf48577a86c5af27407b1e366f341d8e",
                "night_young_hypno.csv": "md5:0e6de9291533623f5345ddcf4cce6183",
                "resting_EO_200Hz_raw.fif": "md5:fac1a071930b10ffef4b197880b39dfb",
                "sub-02_hypno_30s.txt": "md5:bcf75ac8180cdc68c5a6d20f921d1473",
                "sub-02_mne_raw.fif": "md5:eb9360c29f21092834bfd67079b3c57e",
            },
        },
    },
}


def _init_repository(name, version):
    """
    Create a :py:class:`~pooch.Pooch` instance for a given dataset.
    Populates with available filenames and checksums.

    Cache location defaults to ``pooch.os_cache("yasa")`` and can be
    overwritten with the ``YASA_DATA_DIR`` environment variable.

    Parameters
    ----------
    name : str
        The name of the dataset.
    version : str
        The version string of the dataset.

    Returns
    -------
    repo : :py:class:`pooch.Pooch`
        The :py:class:`~pooch.Pooch` instance that can :py:meth:`~pooch.Pooch.fetch`
        from the dataset.

    Examples
    --------
    >>> from pprint import pprint
    >>> import yasa
    >>> repo = yasa.fetchers._init_repository("sample", version="v1")
    >>> # Print the number of available files
    >>> print(len(repo.registry))
    13
    >>> print(sorted(repo.registry_files)[:3])
    ['ECG_8hrs_200Hz.npz', 'EOGs_REM_256Hz.npz', 'N2_spindles_15sec_200Hz.txt']
    >>> pprint({k: v for k, v in repo.registry.items() if k in sorted(repo.registry_files)[:3]})
    {'ECG_8hrs_200Hz.npz': 'md5:3a05f39925a009e04f0c0c88da4f429e',
    'EOGs_REM_256Hz.npz': 'md5:ef86dbef2d6ad99d54aad275175de8e1',
    'N2_spindles_15sec_200Hz.txt': 'md5:6601dad681120d3a9ff8ce4f9a9042a0'}
    """
    default_cache_dir = pooch.os_cache(__package__)  # ~/local/caches/yasa
    cache_dir_env_var = f"{__package__}_DATA_DIR".upper()  # YASA_DATA_DIR
    repo = pooch.create(
        path=default_cache_dir,
        base_url=REGISTRY[name][version]["base_url"],
        registry=REGISTRY[name][version]["registry"],
        env=cache_dir_env_var,
    )
    return repo


def fetch_sample(fname, version="v1", **kwargs):
    """
    Download (i.e., :py:meth:`~pooch.Pooch.fetch`) a single file -- _if not already downloaded_ --
    from the YASA samples dataset on `Zenodo <https://doi.org/10.5281/zenodo.14564284>`_.

    This function always returns a filename as a :py:class:`~pathlib.Path` to the local file.
    It will first check for a local copy of the file and download it if not found.

    The default location to store the file is in a ``yasa/`` folder
    in the user's system-dependent cache directory. (``pooch.os_cache("yasa")``)
    If you want to download the file to a different location,
    you can set the ``YASA_DATA_DIR`` environment variable to the desired path.

    Parameters
    ----------
    fname : str
        The name of the file to :py:meth:`~pooch.Pooch.fetch`.
        Must be one of the filenames available in the YASA samples dataset.
        See the `Zenodo repo <https://doi.org/10.5281/zenodo.14564284>`_ for available filenames.

    version : str, optional
        The version string of the dataset to :py:meth:`~pooch.Pooch.fetch`.
        Setting this to ``latest`` (default) is equivalent to setting to the latest version string.
        Must be one of the versions available for the YASA samples dataset.
        See the `Zenodo repo <https://doi.org/10.5281/zenodo.14564284>`_ for available versions.

    **kwargs : dict
        Optional keyword arguments passed to :py:meth:`~pooch.Pooch.fetch`.
        For example, set ``progressbar=True`` to display a progress bar (requires ``tqdm``).

    Returns
    -------
    fpath : :py:class:`pathlib.Path`
        The :py:class:`~pathlib.Path` to the downloaded file.

    Examples
    --------
    >>> import numpy as np
    >>> import yasa
    >>> # Get path to local hypnogram file (will download if not already present)
    >>> fpath = yasa.fetch_sample("night_young_hypno.csv")
    >>> print(fpath.exists())
    True
    >>> # Load the hypnogram
    >>> stages_int = np.loadtxt(fpath, skiprows=1, dtype=int)
    >>> stages_str = yasa.hypno_int_to_str(stages_int)
    >>> hyp = yasa.Hypnogram(stages_str)
    >>> print(hyp.hypno.head(3))
    Epoch
    0    WAKE
    1    WAKE
    2    WAKE
    Name: Stage, dtype: category
    Categories (7, object): ['WAKE', 'N1', 'N2', 'N3', 'REM', 'ART', 'UNS']

    You can also set the ``YASA_DATA_DIR`` environment variable to a custom location.

    >>> import os
    >>> os.environ["YASA_DATA_DIR"] = "~/Desktop/my_yasa_data"
    >>> fpath = yasa.fetch_sample("night_young_hypno.csv")
    """
    allowed_versions = set(REGISTRY["sample"].keys())
    assert isinstance(fname, str), "`fname` must be a string"
    assert isinstance(version, str), "`version` must be a string"
    assert version in allowed_versions, f"`version` must be one of {allowed_versions}."
    pup = _init_repository("sample", version=version)
    fname = pup.fetch(fname, **kwargs)
    fpath = Path(fname)
    return fpath

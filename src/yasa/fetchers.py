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


def _init_doi_repository(doi, populate_registry=True):
    """
    Create a :py:class:`~pooch.Pooch` instance from a repository DOI
    and automatically populate the registry with available filenames and checksums.

    Cache location defaults to ``pooch.os_cache("yasa")`` and can be
    overwritten with the ``YASA_DATA_DIR`` environment variable.

    Parameters
    ----------
    doi : str
        DOI of the repository.
    populate_registry : bool, optional
        If ``True`` (default), the registry will be automatically populated with available
        filenames and checksums.

    Returns
    -------
    repo : :py:class:`pooch.Pooch`
        The :py:class:`~pooch.Pooch` instance that can :py:meth:`~pooch.Pooch.fetch` the dataset.

    Examples
    --------
    >>> from pprint import pprint
    >>> import yasa
    >>> repo = yasa.fetchers._init_doi_repository("10.5281/zenodo.14564284")
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
    doi_url = f"doi:{doi}"
    default_cache_dir = pooch.os_cache(__package__)  # ~/local/caches/yasa
    cache_dir_env_var = f"{__package__}_DATA_DIR".upper()  # YASA_DATA_DIR
    repo = pooch.create(path=default_cache_dir, base_url=doi_url, env=cache_dir_env_var)
    if populate_registry:
        repo.load_registry_from_doi()
    return repo


def fetch_sample(fname, version="v1", **kwargs):
    """
    Download (i.e., :py:meth:`~pooch.Pooch.fetch`) a single file -- _if not already downloaded_ --
    from the YASA samples dataset on `Zenodo <https://doi.org/10.5281/zenodo.14564284>`_.

    This function always returns a filename as a :py:class:`~pathlib.Path` to the local file.
    It will first check for a local copy of the file and download it if not found.

    The default location to store the file is in a ``yasa/`` folder
    in the user's system-dependent cache directory.
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
    repository_dois = {
        "latest": "10.5281/zenodo.14564284",
        "v1": "10.5281/zenodo.14564285",
    }
    assert isinstance(fname, str), "`fname` must be a string"
    assert isinstance(version, str), "`version` must be a string"
    assert version in repository_dois, f"`version` must be one of {list(repository_dois)}."
    doi = repository_dois[version]
    pup = _init_doi_repository(doi, populate_registry=True)
    fname = pup.fetch(fname, **kwargs)
    fpath = Path(fname)
    return fpath

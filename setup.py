#! /usr/bin/env python
#
# Copyright (C) 2018 Raphael Vallat

DESCRIPTION = "YASA: Analysis of polysomnography recordings."
LONG_DESCRIPTION = """YASA (Yet Another Spindle Algorithm) : an open-source Python package to analyze polysomnographic sleep recordings.
"""

DISTNAME = "yasa"
MAINTAINER = "Raphael Vallat"
MAINTAINER_EMAIL = "raphaelvallat9@gmail.com"
URL = "https://github.com/raphaelvallat/yasa/"
LICENSE = "BSD (3-clause)"
DOWNLOAD_URL = "https://github.com/raphaelvallat/yasa/"
VERSION = "0.6.5"
PACKAGE_DATA = {"yasa.data.icons": ["*.svg"]}

INSTALL_REQUIRES = [
    "numpy>=1.18.1",
    "scipy",
    "pandas",
    "matplotlib",
    "seaborn",
    "mne>=1.3",
    "numba>=0.57.1",
    "antropy",
    "scikit-learn",
    "tensorpac>=0.6.5",
    "pyriemann>=0.2.7",
    "sleepecg>=0.5.0",
    "lspopt",
    "ipywidgets",
    "joblib",
    "lightgbm",
]

PACKAGES = [
    "yasa",
]

CLASSIFIERS = [
    "Intended Audience :: Science/Research",
    "Programming Language :: Python :: 3.8",
    "Programming Language :: Python :: 3.9",
    "Programming Language :: Python :: 3.10",
    "Programming Language :: Python :: 3.11",
    "License :: OSI Approved :: BSD License",
    "Operating System :: POSIX",
    "Operating System :: Unix",
    "Operating System :: MacOS",
]

try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

if __name__ == "__main__":
    setup(
        name=DISTNAME,
        author=MAINTAINER,
        author_email=MAINTAINER_EMAIL,
        maintainer=MAINTAINER,
        maintainer_email=MAINTAINER_EMAIL,
        description=DESCRIPTION,
        long_description=LONG_DESCRIPTION,
        license=LICENSE,
        url=URL,
        version=VERSION,
        download_url=DOWNLOAD_URL,
        install_requires=INSTALL_REQUIRES,
        include_package_data=True,
        packages=PACKAGES,
        package_data=PACKAGE_DATA,
        classifiers=CLASSIFIERS,
    )

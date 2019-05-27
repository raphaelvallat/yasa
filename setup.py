#! /usr/bin/env python
#
# Copyright (C) 2018 Raphael Vallat

DESCRIPTION = "Yet Another Spindle Algorithm"
LONG_DESCRIPTION = """YASA (Yet Another Spindle Algorithm) is a fast and data-agnostic sleep spindles / slow-waves detection algorithm.
"""

DISTNAME = 'yasa'
MAINTAINER = 'Raphael Vallat'
MAINTAINER_EMAIL = 'raphaelvallat9@gmail.com'
URL = 'https://github.com/raphaelvallat/yasa/'
LICENSE = 'BSD (3-clause)'
DOWNLOAD_URL = 'https://github.com/raphaelvallat/yasa/'
VERSION = '0.1.4'
PACKAGE_DATA = {'yasa.data.icons': ['*.svg']}

INSTALL_REQUIRES = [
    'numpy>=1.15',
    'scipy>=1.1',
    'pandas>=0.23',
    'matplotlib>=3.0.2',
    'seaborn>=0.9.0',
    'mne>=0.17.0',
    'numba>=0.39.0',
    'scikit-learn>=0.20'
]

PACKAGES = [
    'yasa',
]

CLASSIFIERS = [
    'Intended Audience :: Science/Research',
    'Programming Language :: Python :: 3.6',
    'Programming Language :: Python :: 3.7',
    'License :: OSI Approved :: BSD License',
    'Operating System :: POSIX',
    'Operating System :: Unix',
    'Operating System :: MacOS'
]

try:
    from setuptools import setup
    _has_setuptools = True
except ImportError:
    from distutils.core import setup

if __name__ == "__main__":

    setup(name=DISTNAME,
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

#!/usr/bin/env python

from setuptools import setup, find_packages

setup(name='pyTRACE',
      python_requires='>=3.10',
      version='0.1.0',
      description='Tracer-based Rapid Anthropogenic Carbon Estimation (TRACE) in Python',
      long_description=open('README.md').read(),
      author='Daniel Sandborn & Brendan Carter',
      author_email='sandborn@uw.edu',
      url='https://github.com/d-sandborn/pyTRACE',
      packages=find_packages(),
      install_requires=[
          'numpy',
          'pandas',
          'netcdf4',
          'xarray',
          'scipy',
          'setuptools',
          'matplotlib',
          'pyyaml',
          'tqdm',
          'seawater', #deprecated, kept for comparability with TRACEv1
          'gsw', #replacement for seawater
          'PyCO2SYS @ git+https://github.com/mvdh7/PyCO2SYS@v2.0.0-b4',
          'numba',
          'shapely',
          'geopandas'
      ],
      classifiers=[
          'Programming Language :: Python :: 3.12',
          'Development Status :: 4 - Beta',
          'Intended Audience :: Science/Research',
          'License :: OSI Approved :: MIT License',
          'Operating System :: OS Independent',
          'Topic :: Scientific/Engineering :: Oceanography',
          ],
)

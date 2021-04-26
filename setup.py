from __future__ import absolute_import
from __future__ import print_function

import io
import re
from glob import glob
from os.path import basename
from os.path import dirname
from os.path import join
from os.path import splitext

from setuptools import find_packages
from setuptools import setup


with open("README.md", "r") as fh:
    long_description = fh.read()


def read(*names, **kwargs):
    with io.open(
        join(dirname(__file__), *names),
        encoding=kwargs.get('encoding', 'utf8')
    ) as fh:
        return fh.read()


setup(
    name="LightcurveFitting",
    version="0.1",
    author="Bjorn Ahlgren",
    author_email="bjornah@kth.se",
    license='BSD-2-Clause',
    description="A package to fit a collection of analytical functions to binned time series data",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='',
    packages=find_packages('src'),
    package_dir={'': 'src'},
    py_modules=[splitext(basename(path))[0] for path in glob('src/*.py')],
    include_package_data=True,
    zip_safe=False,
    classifiers=[
        "Programming Language :: Python :: 3.7",
        "License :: OSI Approved :: BSD 3-Clause License",
        "Operating System :: MacOS",
    ],
    python_requires='>=3.5',
)

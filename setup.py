""" Setup for wav2vec2aligner
"""

import datetime as dt
from os import path

from setuptools import find_packages, setup

build_no = dt.datetime.today().strftime("%Y%m%d")

this_directory = path.abspath(path.dirname(__file__))

with open(path.join(this_directory, "requirements.txt"), encoding="utf8") as f:
    REQS = f.read().splitlines()

setup(
    name="ctc-segmentation-aligner",
    python_requires=">=3.8",
    version="1.0",
    author="Aidan Pine",
    author_email="hello@aidanpine.ca",
    license="MIT",
    description="Module for performing zero-shot forced alignment",
    platform=["any"],
    packages=find_packages(),
    include_package_data=True,
    install_requires=REQS,
    entry_points={"console_scripts": ["ctc-segmenter = aligner.cli:app"]},
    zip_safe=False,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)

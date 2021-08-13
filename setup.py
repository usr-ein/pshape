#!/usr/bin/env python3
import pathlib

from setuptools import setup

HERE = pathlib.Path(__file__).parent
README = (HERE / "README.md").read_text()

setup(
    name="pshape",
    version="0.2.1",
    packages=["pshape"],
    description="Prints NumPy-like arrays' shapes, mins, means, and maxes, as well as the names of the input variable outside the functions' scope",
    long_description=README,
    long_description_content_type="text/markdown",
    url="https://github.com/sam1902/pshape",
    author="Samuel Prevost",
    author_email="samuel.prevost@pm.me",
    licence="BSD",
    classifiers=[
        "License :: OSI Approved :: BSD License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
    ],
    include_package_data=True,
    install_requires=['numpy'],
)

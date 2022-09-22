#!/usr/bin/python3
# -*- coding: utf-8 -*-
from setuptools import find_packages
from setuptools import setup

with open("README.rst") as fp:
    readme = fp.read()

with open("HISTORY.rst") as fp:
    history = fp.read()

setup(
    name="pyStrata",
    version="0.5.1",
    description="Site Response Analysis with Python",
    long_description=readme + "\n\n" + history,
    long_description_content_type="text/x-rst",
    author="Albert Kottke",
    author_email="albert.kottke@gmail.com",
    url="http://github.com/arkottke/pystrata",
    packages=find_packages(),
    install_requires=[
        "matplotlib",
        "numpy",
        "pyrvt",
        "pykooh",
        "scipy",
        "setuptools",
        "typing",
    ],
    extras_require={
        "dataframe": ["pandas"],
    },
    keywords="site response",
    license="MIT",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Environment :: Console",
        "License :: OSI Approved :: MIT License",
        "Natural Language :: English",
        "Operating System :: OS Independent",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3.6",
        "Topic :: Scientific/Engineering",
        "Intended Audience :: Science/Research",
    ],
    test_suite="tests",
)

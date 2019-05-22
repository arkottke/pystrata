#!/usr/bin/python3
# -*- coding: utf-8 -*-

from setuptools import setup, find_packages

with open('README.md') as fp:
    readme = fp.read()

with open('HISTORY.md') as fp:
    history = fp.read()

setup(
    name='pySRA',
    version='0.4.1',
    description='Site Response Analysis with Python',
    long_description=readme + '\n\n' + history,
    long_description_content_type='text/markdown',
    author='Albert Kottke',
    author_email='albert.kottke@gmail.com',
    url='http://github.com/arkottke/pysra',
    packages=find_packages(),
    install_requires=[
        'matplotlib',
        'numpy',
        'pyrvt',
        'scipy',
        'setuptools',
        'typing',
    ],
    keywords='site response',
    license='MIT',
    classifiers=[
        'Development Status :: 4 - Beta',
        'Environment :: Console',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Operating System :: OS Independent',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3.6',
        'Topic :: Scientific/Engineering',
        'Intended Audience :: Science/Research',
    ],
    test_suite='tests', )

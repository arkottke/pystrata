#!/usr/bin/env python3
# encoding: utf-8

from setuptools import setup

import versioneer

config = dict(
    name='pysra',
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    description='Site Response Analysis with Python',
    author='Albert Kottke',
    author_email='albert.kottke@gmail.com',
    url='http://github.com/arkottke/pysra',
    entry_points=dict(
    ),
    packages=['pysra'],
    package_data=dict(
        pysra=['data/*']
    ),
    requires=[
        'matplotlib',
        'nose',
        'numpy',
        'pyrvt'
        'scipy',
        'setuptools',
        'typing',
    ],
    test_suite='nose.collector',
    classifiers=[
        'Development Status :: 4 - Beta',
        'Environment :: Console',
        'License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)',
        'Natural Language :: English',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 3',
        'Topic :: Scientific/Engineering',
    ],
)

setup(**config)

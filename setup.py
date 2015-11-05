#!/usr/bin/env python3
# encoding: utf-8

from setuptools import setup

config = {
    'name': 'pysra',
    'version': '0.1',
    'description': 'Site Response Analysis with Python',
    'author': 'Albert Kottke',
    'author_email': 'albert.kottke@gmail.com',
    'url': 'http://github.com/arkottke/pysra',
    'entry_points': {
        'console_scripts': [
            ],
        },
    'packages': ['pysra'],
    'package_data': {
        'pysra': ['data/*']
    },
    'requires': [
        'matplotlib',
        'nose',
        'numpy',
        'scipy',
        'setuptools',
        'typing',
        'pyrvt'
    ],
    'test_suite': 'nose.collector',
    'classifiers': [
        'Development Status :: 4 - Beta',
        'Environment :: Console',
        'License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)',
        'Natural Language :: English',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3',
        'Topic :: Scientific/Engineering',
    ],
}

setup(**config)

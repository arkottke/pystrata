#!/usr/bin/env python
# encoding: utf-8

try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup


with open('README.rst') as readme_file:
    readme = readme_file.read()

with open('HISTORY.rst') as history_file:
    history = history_file.read()

requirements = [
    'matplotlib',
    'numpy',
]

setup_requirements = [
    'pytest-runner',
]

test_requirements = [
    'pytest >= 2.9.0',
    'pytest-cov',
    'pytest-flake8'
]

setup(
    name='pySRA',
    version='0.0.1.dev0',
    description="Site response analyses implemented in Python.",
    long_description=readme + '\n\n' + history,
    author="Albert Kottke",
    author_email='albert.kottke@gmail.com',
    url='https://github.com/arkottke/pysra',
    packages=[
        'pysra'
    ],
    package_dir={'pysra':
                 'pysra'},
    package_data={'pysra': ['data/*.csv', 'data/*.json']},
    include_package_data=True,
    install_requires=requirements,
    license="MIT",
    zip_safe=False,
    keywords='pysra',
    classifiers=[
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python',
        'Topic :: Scientific/Engineering',
        'Intended Audience :: Science/Research',
    ],
    setup_requires=setup_requirements,
    tests_require=test_requirements,
    test_suite='tests',
)

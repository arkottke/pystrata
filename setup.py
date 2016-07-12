#!/usr/bin/env python3
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
    'nose',
    'numpy',
    'pyrvt',
    'scipy',
    'setuptools',
    'typing',
]

requirements_test = [

]

setup(
    name='pySRA',
    version='0.1.0',
    description='Site Response Analysis with Python',
    author='Albert Kottke',
    author_email='albert.kottke@gmail.com',
    url='http://github.com/arkottke/pysra',
    license='MIT',
    entry_points=dict(
    ),
    packages=['pysra'],
    package_data=dict(
        pysra=['data/*']
    ),
    install_requires=requirements,
    keywords='site response',
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
    test_suite='tests',
    tests_requirements=requirements_test,
)
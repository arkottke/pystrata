#!/usr/bin/python3
# -*- coding: utf-8 -*-

<<<<<<< HEAD
from setuptools import setup, find_packages

=======
>>>>>>> 4d71de0126fd2979338192782a55642ad41b2c46
with open('README.rst') as readme_file:
    readme = readme_file.read()

with open('HISTORY.rst') as history_file:
    history = history_file.read()

<<<<<<< HEAD
=======
requirements = [
    'matplotlib',
    'numpy',
    'pyrvt',
    'scipy',
    'setuptools',
    'typing',
]

requirements_test = []

>>>>>>> 4d71de0126fd2979338192782a55642ad41b2c46
setup(
    name='pySRA',
    version='0.1.0',
    description='Site Response Analysis with Python',
    long_description=readme + '\n\n' + history,
    author='Albert Kottke',
    author_email='albert.kottke@gmail.com',
    url='http://github.com/arkottke/pysra',
<<<<<<< HEAD
    packages=find_packages(),
    install_requires=[
        'matplotlib',
        'numpy',
        'pyrvt',
        'scipy',
        'setuptools',
        'typing',
    ],
=======
    license='MIT',
    entry_points=dict(),
    packages=['pysra'],
    package_data=dict(pysra=['data/*']),
    install_requires=requirements,
>>>>>>> 4d71de0126fd2979338192782a55642ad41b2c46
    keywords='site response',
    license='MIT',
    classifiers=[
        'Development Status :: 4 - Beta',
<<<<<<< HEAD
=======
        'Environment :: Console',
>>>>>>> 4d71de0126fd2979338192782a55642ad41b2c46
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Operating System :: OS Independent',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3.6',
        'Topic :: Scientific/Engineering',
        'Intended Audience :: Science/Research',
    ],
<<<<<<< HEAD
    test_suite='tests', )
=======
    test_suite='tests',
    tests_requirements=requirements_test, )
>>>>>>> 4d71de0126fd2979338192782a55642ad41b2c46

#!/usr/bin/env python
try:
    from setuptools import find_packages, setup
except ImportError:
    from distutils.core import setup

setup(
    name="gibby",
    version="0.0",
    description="tools for numerical Hessians / Gibbs corrections ",
    author="Brook Wander",
    author_email="bwander@andrew.cmu.edu",
    url="tbd",
    packages=find_packages(),
)

# Copyright (c) 2025-2026, Haopeng Li

from setuptools import setup, find_packages

setup(
    name="piecewise_attn",
    version="0.1.0",
    author='Haopeng Li',
    packages=find_packages(),
    install_requires=[
        "torch>=2.7.1",
        "triton>=3.5.1"
    ],
)
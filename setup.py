# -*- coding: utf-8 -*-
"""
Created on Thu Jul 21 13:52:13 2022

@author: Wuestney

This setup file is based on the template accessed
at https://github.com/pypa/sampleproject on 7/21/2022.
"""

from setuptools import setup, find_packages
import pathlib

here = pathlib.Path(__file__).resolve().parent

long_description = (here / "README.md").read_text(encoding="utf-8")

setup(
      name = "pyusm",
      version = "0.1.0",
      description = "A Python implementation of Approximate Entropy and Sample Entropy for discrete-valued time series.",
      long_description = long_description,
      long_description_content_type = "text/markdown",
      url = "https://github.com/katherine983/discreteMSE",
      author = "Katherine Wuestney",
      # author_email = "katherineann983@gmail.com",
      keywords = "ApEn, SampEn, sample entropy, MSE, categorical time series, discrete-valued time series",
      install_requires = ['numpy>=1.20.1'],
      packages = ['discreteMSE'],
      python_requires = ">=3.6")
"""
locateAnythingForMe - 基于 Eagle 的视觉定位 API 封装
"""
from setuptools import setup, find_packages

setup(
    name="locateAnythingForMe",
    version="0.1.0",
    description="Easy-to-use API wrapper for NVlabs/Eagle multimodal vision model",
    author="",
    packages=find_packages(exclude=["submodules", "scripts", "tests"]),
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.0.0",
        "transformers>=4.36.0",
        "Pillow>=9.0.0",
        "numpy>=1.24.0",
        "accelerate>=0.20.0",
    ],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
)

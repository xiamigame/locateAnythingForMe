"""
locateAnythingForMe - 基于 LocateAnything-3B 的视觉定位 API 封装
"""
from setuptools import setup, find_packages

setup(
    name="locateAnythingForMe",
    version="0.1.0",
    description="Easy-to-use API wrapper for NVlabs/Eagle Embodied (LocateAnything-3B)",
    packages=find_packages(exclude=["submodules", "scripts", "tests"]),
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.0.0",
        "transformers>=4.57.1",
        "accelerate>=1.5.2",
        "Pillow>=9.0.0",
    ],
)

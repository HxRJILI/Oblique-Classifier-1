"""
Setup script for OC1 package.
"""

from setuptools import setup, find_packages

setup(
    name="oc1",
    version="0.1.0",
    description="OC1 Oblique Decision Tree Implementation",
    long_description=open("oc1/README.md").read(),
    long_description_content_type="text/markdown",
    author="OC1 Implementation Team",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.20.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
        ],
        "viz": [
            "matplotlib>=3.3.0",
            "networkx>=2.5",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)

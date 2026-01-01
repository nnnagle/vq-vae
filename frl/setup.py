from setuptools import setup, find_packages

setup(
    name="frl",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "zarr",
        "torch",
        "pyyaml",
        "xarray",
        "rioxarray",
    ],
    python_requires=">=3.8",
)

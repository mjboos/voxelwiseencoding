from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="voxelwiseencoding",
    version="0.0.1",
    author="Moritz Boos",
    author_email="moritz.boos@gmail.com",
    description="A library for training voxelwise encoding models.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/mjboos/voxelwiseencoding",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering"
    ],
    python_requires='>=3.5',
    install_requires=["numpy", "nilearn", "nibabel", "scikit-learn", "scikit-image", "joblib"]
)
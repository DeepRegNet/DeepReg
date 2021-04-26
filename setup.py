import sys

from setuptools import find_packages, setup

# package requirements
if sys.version_info[:2] < (3, 6):
    raise RuntimeError("Python version >= 3.6 required.")
with open("requirements.txt", "r") as f:
    requirements = f.read().splitlines()
if sys.version_info[:2] == (3, 6):
    requirements = [x for x in requirements if "pandas" not in x]
    requirements.append("pandas==1.1.5")
    requirements = [x for x in requirements if "scipy" not in x]
    requirements.append("scipy==1.5.4")
    requirements = [x for x in requirements if "matplotlib" not in x]
    requirements.append("matplotlib==3.3.4")

with open("README.md", "r") as f:
    long_description = f.read()

setup(
    name="deepreg",
    packages=find_packages(exclude=["test", "test.unit", "test.output"]),
    include_package_data=True,
    version="0.0.0",
    license="apache-2.0",
    description="DeepReg is a freely available, "
    "community-supported open-source toolkit for research and education"
    " in medical image registration using deep learning.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="DeepReg Development Team and Community",
    author_email="deepregnet@gmail.com",
    url="http://deepreg.net/",
    download_url="",
    keywords=[
        "Deep Learning",
        "Image Fusion",
        "Medical Image Registration",
        "Neural Networks",
    ],
    zip_safe=False,
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "deepreg_train=deepreg.train:main",
            "deepreg_predict=deepreg.predict:main",
            "deepreg_warp=deepreg.warp:main",
            "deepreg_vis=deepreg.vis:main",
            "deepreg_download=deepreg.download:main",
        ]
    },
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Environment :: Console",
        "Environment :: GPU",
        "Environment :: MacOS X",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Natural Language :: English",
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: Unix",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
        "Topic :: Software Development",
    ],
)

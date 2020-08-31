from setuptools import find_packages, setup

with open("requirements.txt") as f:
    requirements = f.read().splitlines()

with open("README.md", "r") as f:
    long_description = f.read()

setup(
    name="deepreg",
    packages=find_packages(exclude=["test", "test.unit"]),
    package_data={"deepreg": ["config/*.yaml", "config/test/*.yaml"]},
    include_package_data=True,
    version="0.1.0b1",
    license="apache-2.0",
    description="DeepReg is a freely available, community-supported open-source toolkit for research and education in medical image registration using deep learning.",
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
        ]
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Environment :: Console",
        "Environment :: GPU",
        "Environment :: MacOS X",
        "Environment :: Win32 (MS Windows)",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Natural Language :: English",
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: Microsoft :: Windows :: Windows 10",
        "Operating System :: Unix",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
        "Topic :: Software Development",
    ],
)

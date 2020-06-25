from setuptools import setup

setup(
    name="deepreg",
    version="0.1.4",
    description="Registration with Deep Learning",
    author="Yunguan Fu",
    packages=["deepreg"],
    zip_safe=False,
    install_requires=[
        "h5py",
        "numpy>=1.16",
        "nibabel",
        "pyyaml",
        "matplotlib",
        "argparse",
        "tqdm",
        "tensorflow==2.2",
        "pytest>=4.6",
        "pytest-cov",
        "pytest-dependency",
        "black",
        "flake8",
        "pre-commit",
    ],
    entry_points={
        "console_scripts": [
            "train=deepreg.train:main",
            "predict=deepreg.predict:main",
            "gen_tfrecord=deepreg.gen_tfrecord:main",
        ]
    },
)

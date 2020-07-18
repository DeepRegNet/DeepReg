from setuptools import setup

setup(
    name="deepreg",
    version="0.1.5",
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
        "pre-commit",
        "seed-isort-config",
        "isort",
        "black",
        "flake8",
        "simple_http_server",
        "testfixtures",
    ],
    entry_points={
        "console_scripts": ["train=deepreg.train:main", "predict=deepreg.predict:main"]
    },
)

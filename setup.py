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
        "pandas",
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
        "testfixtures",
        "notebook",
        # dev
        "simple_http_server",
        "sphinx",
        "sphinx_rtd_theme",
    ],
    entry_points={
        "console_scripts": [
            "deepreg_train=deepreg.train:main",
            "deepreg_predict=deepreg.predict:main",
            "deepreg_warp=deepreg.warp:main",
        ]
    },
)

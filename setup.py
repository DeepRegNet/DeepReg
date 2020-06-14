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
        "numpy",
        "nibabel",
        "pyyaml",
        "matplotlib",
        "click",
        "tqdm",
        "tensorflow==2.2",
        "nilearn",
        "pytest",
    ],
    entry_points={
        "console_scripts": [
            "train=deepreg.train:main",
            "predict=deepreg.predict:main",
            "gen_tfrecord=deepreg.gen_tfrecord:main",
        ]
    },
)

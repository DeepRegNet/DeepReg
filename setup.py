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
    ],
    entry_points={
        "console_scripts": [
            "deepreg_train=deepreg.train:main",
            "deepreg_predict=deepreg.predict:main",
        ]
    },
)

from setuptools import find_packages, setup

with open("requirements.txt") as f:
    requirements = f.read().splitlines()

setup(
    name="deepreg",
    version="0.0.1",
    description="Registration with Deep Learning",
    author="DeepReg",
    packages=find_packages(),
    zip_safe=False,
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "deepreg_train=deepreg.train:main",
            "deepreg_predict=deepreg.predict:main",
            "deepreg_warp=deepreg.warp:main",
        ]
    },
)

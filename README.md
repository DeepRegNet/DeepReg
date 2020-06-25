<img src="./deepreg_logo_purple.svg" alt="deepreg_logo" title="DeepReg" width="150" />  

# DeepReg: deep learning for image registration

DeepReg is an open-source toolkit for research in medical image registration using deep learning. The current version is based on TensorFlow 2. This toolkit contains implementations for unsupervised- and weaky-supervised algorithms with their combinations and variants, with a practical focus on diverse clinical applications, as in provided examples.

This is still under development. However some of the functionalities can be accessed.



## Quick start
- Create a new virtual environment using Anaconda/Miniconda:  
`conda create --name deepreg python=3.7 tensorflow-gpu=2.2`

- Install DeepReg:  
`pip install -e .`

- Train a registration network using test data:  
`train --gpu <str> --config_path <str> --gpu_allow_growth --ckpt_path <str> --log <str>`

- Prediction using a trained registration network:  
`predict --gpu <str> --mode <str> --ckpt_path <str> --gpu_allow_growth --log <str> --batch_size <int> --sample_label <str>` 


## Tutorials
### Two ways to get started with DeepReg  

[Get started with image registration using deep learning](https://github.com/ucl-candi/DeepReg/blob/master/tutorials/registration.md)  

[Get started with demos](https://github.com/ucl-candi/DeepReg/blob/master/tutorials/demos.md)  

### Other topics 
[How to arrange data files and folders to use predefined data loaders](https://github.com/ucl-candi/DeepReg/blob/master/tutorials/predefined_loader.md)  

[Training data sampling options](https://github.com/ucl-candi/DeepReg/blob/master/tutorials/sampling.md)  

(under development)

[Configuration options](https://github.com/ucl-candi/DeepReg/blob/master/tutorials/configuration.md)  


......

### Setup
(under development)

## Demos
(under development)


## Contributions
We welcome contributions! Please refer to the [contribution guidelines](https://github.com/ucl-candi/DeepReg/blob/master/docs/CONTRIBUTING.md) for the toolkit.
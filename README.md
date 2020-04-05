# Deep Registration


## Installation

The easiest way to install the python environment is with Miniconda/Anaconda.
- Use `conda create --name py36-tf2 python=3.6 tensorflow-gpu` to setup the latest gpu-support with tf.
- Use `pip install -r requirements.txt` to install the rest requirements

## Demo

### Train

The training script is `train.py` and it accepts the following parameters
- `-g` or `--gpu`, **required**, providing available GPU indices, e.g. `-g 0` uses GPU of index 0 and `-g "0,1"` uses GPU of index 0 and 1.
- `-c` or `--config`, **required**, providing the path of the configuration file, `-c demo.yaml`. Some default configuration files are provided under `src/config/`.
- `-m` or `--memory`, providing this flag will prevent tensorflow to reserve all available GPU memory.
- `--ckpt`, providing the checkpoint to load, to prevent start training from random initialization. The path must ended in `.ckpt`, e.g. `--ckpt logs/demo/save/weights-epoch100.ckpt`
- `-l` or `--log`, providing the name of log folder. It not provided, a timestamp based folder name will be used.

For example, `python train.py -c demo.yaml -l demo -g "0,1"` will launch a training using the configuration file `demo.yaml` with GPU of index 0 and 1 and the log will be saved under `logs/demo`.

### Predict

The prediction script is `predict.py` and it accepts the following parameters
- `-g` or `--gpu`, **required**, providing available GPU indices, e.g. `-g 0` uses GPU of index 0. Multi-GPU setting is not tested for prediction.
- `--mode`, **required**, providing which part of the data should be evaluated. Must be one of `train`/`valid`/`test`.
- `--ckpt`, **required**, providing the checkpoint to load, to prevent start training from random initialization. The path must ended in `.ckpt`, e.g. `--ckpt logs/demo/save/weights-epoch100.ckpt`. When loading the checkpoint, a backup configuration file wil be used, in the example above, the configuration file will be under `logs/demo/`. 
- `-m` or `--memory`, providing this flag will prevent tensorflow to reserve all available GPU memory.
- `--bs`, providing the batch size, the number of data samples must be divided evenly by batch size during prediction. If not provided, batch size of 1 will be used.


For example, `python predict.py -g 0 --ckpt logs/demo/save/weights-epoch100.ckpt` will launch a prediction using the provided checkpoint with one GPU. The results will be saved under `logs` in a new folder, in which a `metric.log` will be generated. An example of `metric.log` will be like

```
image 0, label 0, dice 0.7821, dist 6.9741
image 0, label 1, dice 0.0147, dist 7.1142
image 0, label 2, dice 0.0000, dist 13.4373
image 0, label 3, dice 0.0000, dist 14.0168
image 0, label 4, dice 0.0000, dist 14.0412
image 0, label 5, dice 0.0000, dist 18.5782
image 1, label 0, dice 0.8591, dist 1.4560
image 1, label 1, dice 0.0000, dist 9.1976
image 1, label 2, dice 0.0000, dist 8.4542
image 1, label 3, dice 0.0000, dist 16.8208
image 1, label 4, dice 0.0000, dist 5.3310
image 1, label 5, dice 0.0000, dist 27.9396

```

## Development

### Data

To use this package with a custom dataset,
1. Create a folder under `deepreg/data/`, e.g. `deepreg/data/custom/`.

2. Create a new sample configuration file under `deepreg/config/`, e.g. `deepreg/config/custom.yaml`.

   There is no need to change the `tf` part, and all data related configurations should be under `data` part 
   and the only required attribute is `name`. 

3. Write a new data loader to load the custom data.

    Each data sample consists of `((moving_image`, `fixed_image`, `moving_label`, `indices`), `fixed_image)` where
    
    - images and labels are all assumed to be 3D single-channel images and of shape `[dim1, dim2, dim3]`.
    - indices are of shape `[num_indices]`, used for identifying the data sample. 

    An interface of using generator to build TF dataset is provide
    in `deepreg/data/loader_gen.py` there are three functions need to be implemented
    - `get_generator` which builds a generator to return data. This function is used for data loading.
    - `split_indices` which splits the indices into `image_index` and `label_index`,
    where `label_index` must be a integer and `image_index` can be an integer or a tuple. This function is used for prediction.
    - `image_index_to_dir` which format the image index into a string. This function is used for prediction.
    
4. Write a new load function `get_data_loader` which returns the data loader given the configuration and the mode (`train`/`valid`/`test`).

   Call `get_data_loader` inside `deepreg/data/load.py` for the custom data.

### Model
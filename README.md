# Deep Registration


## Environment Setup

The easiest way to install the python environment is with Miniconda/Anaconda.
- Use `conda create --name py36-tf2 python=3.6 tensorflow-gpu` to setup the latest gpu-support with tf.
- Use `pip install -r requirements.txt` to install the rest requirements

## Train

The training script is `train.py` and it accepts the following parameters
- `-g` or `--gpu`, **required**, providing available GPU indices, e.g. `-g 0` uses GPU of index 0 and `-g "0,1"` uses GPU of index 0 and 1.
- `-c` or `--config`, **required**, providing the path of the configuration file, `-c demo.yaml`. Some default configuration files are provided under `src/config/`.
- `-m` or `--memory`, providing this flag will prevent tensorflow to reserve all available GPU memory.
- `--ckpt`, providing the checkpoint to load, to prevent start training from random initialization. The path must ended in `.ckpt`, e.g. `--ckpt logs/demo/save/weights-epoch100.ckpt`
- `-l` or `--log`, providing the name of log folder. It not provided, a timestamp based folder name will be used.

For example, `python train.py -c demo.yaml -l demo -g "0,1"` will launch a training using the configuration file `demo.yaml` with GPU of index 0 and 1 and the log will be saved under `logs/demo`.

## Predict

The prediction script is `predict.py` and it accepts the following parameters
- `-g` or `--gpu`, **required**, providing available GPU indices, e.g. `-g 0` uses GPU of index 0. Multi-GPU setting is not tested for prediction.
- `--ckpt`, **required**, providing the checkpoint to load, to prevent start training from random initialization. The path must ended in `.ckpt`, e.g. `--ckpt logs/demo/save/weights-epoch100.ckpt`. When loading the checkpoint, a backup configuration file wil be used, in the example above, the configuration file will be under `logs/demo/`. 
- `-m` or `--memory`, providing this flag will prevent tensorflow to reserve all available GPU memory.
- `--bs`, providing the batch size, the number of data samples must be divided evenly by batch size during prediction. If not provided, batch size of 1 will be used.
- `--train`, providing this flag will execute prediction on the training data.

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


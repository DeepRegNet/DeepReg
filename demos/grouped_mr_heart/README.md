# Pairwise registration for grouped cardiac MR images

> **Note**: Please read the
> [DeepReg Demo Disclaimer](introduction.html#demo-disclaimer).

[Source Code](https://github.com/DeepRegNet/DeepReg/tree/main/demos/grouped_mr_heart)

This demo uses the grouped dataset loader to register intra-subject multi-sequence
cardiac magnetic resonance (CMR) images.

## Author

DeepReg Development Team

## Application

Computer-assisted management for patients suffering from myocardial infraction (MI)
often requires quantifying the difference and comprising the multiple sequences, such as
the late gadolinium enhancement (LGE) CMR sequence MI, the T2-weighted CMR. They
collectively provide radiological information otherwise unavailable during clinical
practice.

## Data

This demo uses CMR images from 45 patients, acquired from the
[MyoPS2020](http://www.sdspeople.fudan.edu.cn/zhuangxiahai/0/MyoPS20/) challenge held in
conjunction with MICCAI 2020.

## Instruction

### Installation

Please install DeepReg following the [instructions](../getting_started/install.html) and
change the current directory to the root directory of DeepReg project, i.e. `DeepReg/`.

### Download data

Please execute the following command to download the data and pre-trained model. Images
are re-sampled to an isotropic voxel size.

```bash
python demos/grouped_mr_heart/demo_data.py
```

### Launch demo training

Please execute the following command to launch a demo training (the first of the ten
runs of a 9-fold cross-validation). The training logs will be saved under
`demos/grouped_mr_heart/logs_train`, where the saved checkpoints can be used for
prediction later.

```bash
python demos/grouped_mr_heart/demo_train.py
```

Here the training is launched using the GPU of index 0 with a limited number of steps
and reduced size. Please add flag `--no-test` to use the original training
configuration, such as

```bash
python demos/grouped_mr_heart/demo_train.py --no-test
```

### Launch prediction

Please execute the following command to launch the prediction with pre-trained model.
The prediction logs will be saved under `demos/grouped_mr_heart/logs_predict`, where the
visualization of predictions are saved. Check the [CLI documentation](../docs/cli.html)
for more details about prediction output.

```bash
python demos/grouped_mr_heart/demo_predict.py
```

Optionally, the user-trained model can be used by changing the `ckpt_path` variable
inside `demo_predict.py`. Note that the path should end with `.ckpt` and checkpoints are
saved under `logs_train` as mentioned above.

## Contact

Please [raise an issue](https://github.com/DeepRegNet/DeepReg/issues/new/choose) for any
questions.

## Reference

[1] Xiahai Zhuang: Multivariate mixture model for myocardial segmentation combining
multi-source images. IEEE Transactions on Pattern Analysis and Machine Intelligence (T
PAMI), vol. 41, no. 12, 2933-2946, Dec 2019. link.

[2] Xiahai Zhuang: Multivariate mixture model for cardiac segmentation from
multi-sequence MRI. International Conference on Medical Image Computing and
Computer-Assisted Intervention, pp.581-588, 2016.

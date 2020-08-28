# Pairwise registration for grouped multi-sequence cardiac MR images

> **Note**: Please read the
> [DeepReg Demo Disclaimer](introduction.html#demo-disclaimer).

This demo uses the grouped dataset loader to register intra-subject multi-sequence
cardiac magnetic resonance (CMR) images.

## Author

DeepReg Development Team (raise an issue:
https://github.com/DeepRegNet/DeepReg/issues/new, or mailto the author:
yipeng.hu@ucl.ac.uk)

## Application

Computer-assisted management for patients suffering from myocardial infraction (MI)
often requires quantifying the difference and comprising the multiple sequences, such as
he late gadolinium enhancement (LGE) CMR sequence MI, the T2-weighted CMR. The
collectively provide radiological information otherwise unavailable during clinical
practice.

## Instruction

- [Install DeepReg](https://deepregnet.github.io/DeepReg/#/quick_start?id=install-the-package);
- Change current directory to the root directory of DeepReg project;
- Run [demo_data.py](./demo_data.py) script to download all the CMR dataset in a zip
  file. The script also splits the data into train, val and test sets re-samples all the
  images to an isotropic voxel size.

```bash
python demos/grouped_mr_heart/demo_data.py
```

- Call `deepreg_train` from command line. The following example uses a single GPU and
  launches the first of the ten runs of a 9-fold cross-validation, as specified in the
  [`dataset` section](./grouped_mr_heart_dataset0.yaml) and the
  [`train` section](./grouped_mr_heart_train.yaml), which can be specified in
  [seperate yaml files](https://deepregnet.github.io/DeepReg/#/tutorial_experiment?id=cross-validation);

```bash
deepreg_train --gpu "0" --config_path demos/grouped_mr_heart/grouped_mr_heart.yaml --log_dir grouped_mr_heart
```

- Call `deepreg_predict` from command line to use the saved ckpt file for testing on the
  data partitions specified in the config file, a copy of which will be saved in the
  [log_dir]. The following example uses a pre-trained model, on CPU. If not specified,
  the results will be saves at the created timestamp-named directories under /logs.

```bash
deepreg_predict --gpu "" --config_path demos/grouped_mr_heart/grouped_mr_heart.yaml --ckpt_path demos/grouped_mr_heart/dataset/pre-trained/weights-epoch500.ckpt --save_png --mode test
```

## Pre-trained Model

A pre-trained model will be downloaded after running [demo_data.py](./demo_data.py) and
unzipped at dataset folder under the demo folder. This pre-trained model will be used by
default with `deepreg_predict`. Run the user-trained model by specify `--ckpt_path` to
where the ckpt files are save, in this case (specified by `deepreg_train` as above),
/logs/grouped_mr_heart/.

## Data

This demo uses CMR images from 45 patients, acquired from the
[MyoPS2020](http://www.sdspeople.fudan.edu.cn/zhuangxiahai/0/MyoPS20/) challenge held in
conjunction with MICCAI 2020.

## Tested DeepReg Tag

#74e7b1f749d0df1c140494eba0204f0edd1d7b1e

## References

[1] Xiahai Zhuang: Multivariate mixture model for myocardial segmentation combining
multi-source images. IEEE Transactions on Pattern Analysis and Machine Intelligence (T
PAMI), vol. 41, no. 12, 2933-2946, Dec 2019. link. [2] Xiahai Zhuang: Multivariate
mixture model for cardiac segmentation from multi-sequence MRI. International Conference
on Medical Image Computing and Computer-Assisted Intervention, pp.581-588, 2016.

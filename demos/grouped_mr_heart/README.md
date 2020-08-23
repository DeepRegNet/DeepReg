# Pairwise registration for grouped multi-sequence cardiac MR images

This demo uses the grouped dataset loader to register intra-subject multi-sequence
cardiac magnetic resonance (CMR) images.

## Author

DeepReg Development Team (correspondence: yipeng.hu@ucl.ac.uk)

## Application

Computer-assisted management for patients suffering from myocardial infraction (MI)
often requires quantifying the difference and compariring the multiple sequences, such
as he late gadolinium enhancement (LGE) CMR sequence MI, the T2-weighted CMR. The
collectively provide radiological information otherwise unavailable duing clinical
practice.

## Instruction

- [Install DeepReg](https://deepregnet.github.io/DeepReg/#/quick_start?id=install-the-package);
- Change current directory to the root directory of DeepReg project;
- Run [demo_data.py](./demo_data.py) script to download example 10 folds of unpaired 3D
  ultrasound images and the pre-trained model.

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
  data partitions specified in the config file, a copy of which woule be saved in the
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

This demo uses CMR images from 45 patietns, acquired from the
[MyoPS2020](http://www.sdspeople.fudan.edu.cn/zhuangxiahai/0/MyoPS20/) challenge held in
conjunction with MICCAI 2020.

## Tested DeepReg Tag

tbc

## References

[1] Xiahai Zhuang: Multivariate mixture model for myocardial segmentation combining
multi-source images. IEEE Transactions on Pattern Analysis and Machine Intelligence (T
PAMI), vol. 41, no. 12, 2933-2946, Dec 2019. link. [2] Xiahai Zhuang: Multivariate
mixture model for cardiac segmentation from multi-sequence MRI. International Conference
on Medical Image Computing and Computer-Assisted Intervention, pp.581-588, 2016.

# Unpaired abdominal CT image registration

> **Note**: Please read the
> [DeepReg Demo Disclaimer](introduction.html#demo-disclaimer).

## Author

DeepReg Development Team (raise an issue:
https://github.com/DeepRegNet/DeepReg/issues/new, or mailto the author:
e.bonmati@ucl.ac.uk)

## Application

This demo shows how to register unpaired abdominal CT data from different patients using
DeepReg. In addition, the demo demonstrates the difference betwwen the unsupervised,
weakly-supervised and their combination, using a U-Net.

## Data

The data set is from the MICCAI Learn2Reg grand challenge
(https://learn2reg.grand-challenge.org/) task 3 [1], and can be downloaded directly from
https://learn2reg.grand-challenge.org/Datasets/.

## Instructions

- [Install DeepReg](https://deepreg.readthedocs.io/en/latest/getting_started/install.html);

- Change the working directory to the root directory DeepReg

- Run [demo_data.py] to download and extract all files, and to split the data into
  training, validation and testing. If the data has already been downloaded. This will
  also download the pre-trained models

```bash
python ./demos/unpaired_ct_abdomen/demo_data.py
```

After running the command you will have the following directories in
DeepReg/demos/unpaired_ct_abdomen/dataset:

```bash
pre-trained  test  train  val
```

- The next step is to train the network using DeepReg. To train the network, run one of
  the following commands in command line:

- Unsupervised learning

```bash
deepreg_train --gpu "0" --config_path demos/unpaired_ct_abdomen/unpaired_ct_abdomen_unsup.yaml --log_dir unpaired_ct_abdomen_unsup
```

- Weakly-supervised learning

```bash
deepreg_train --gpu "1" --config_path demos/unpaired_ct_abdomen/unpaired_ct_abdomen_weakly.yaml --log_dir unpaired_ct_abdomen_weakly
```

- Combined learning

```bash
deepreg_train --gpu "2" --config_path demos/unpaired_ct_abdomen/unpaired_ct_abdomen_comb.yaml --log_dir unpaired_ct_abdomen_comb
```

- After training the network, run `demo_predict`:

The following example uses a pre-trained model, on CPU.

```bash
deepreg_predict --gpu "" --config_path demos/unpaired_ct_abdomen/unpaired_ct_abdomen_unsup.yaml --ckpt_path demos/unpaired_ct_abdomen/dataset/pre-trained/unsup/weights-epoch5000.ckpt --log_dir unpaired_ct_abdomen_unsup --save_png --mode test
```

```bash
deepreg_predict --gpu "" --config_path demos/unpaired_ct_abdomen/unpaired_ct_abdomen_weakly.yaml --ckpt_path demos/unpaired_ct_abdomen/dataset/pre-trained/weakly/weights-epoch2250.ckpt --log_dir unpaired_ct_abdomen_weakly --save_png --mode test
```

```bash
deepreg_predict --gpu "" --config_path demos/unpaired_ct_abdomen/unpaired_ct_abdomen_comb.yaml --ckpt_path demos/unpaired_ct_abdomen/dataset/pre-trained/comb/weights-epoch2000.ckpt --log_dir unpaired_ct_abdomen_comb --save_png --mode test
```

- Finally, prediction results can be seen in the respective test folders specified in
  `deepreg_predict`.

## Pre-trained Model

Three pre-trained models are available for this demo, for different training strategies
described above. These will be downloaded using the [demo_data.py].

## Tested DeepReg Version

Last commit at which demo was tested: 3157f880eb99ce10fc3a4a8ebcc595bd67be24e1

## Contact

Please raise an issue following the guidelines
(https://deepreg.readthedocs.io/en/325-improve-contributing-pages/contributing/issue.html)

## References

[1] Adrian Dalca, Yipeng Hu, Tom Vercauteren, Mattias Heinrich, Lasse Hansen, Marc
Modat, Bob de Vos, Yiming Xiao, Hassan Rivaz, Matthieu Chabanas, Ingerid Reinertsen,
Bennett Landman, Jorge Cardoso, Bram van Ginneken, Alessa Hering, and Keelin Murphy.
(2020, March 19). Learn2Reg - The Challenge. Zenodo.
http://doi.org/10.5281/zenodo.3715652

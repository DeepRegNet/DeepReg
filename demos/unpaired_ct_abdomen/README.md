# Unpaired CT Abdominal Registration

## Author

Ester Bonmati (e.bonmati@ucl.ac.uk)

## Application

This demo shows how to register unpaired abdominal CT data from different patients using
DeepReg. In addition, the demo demonstrates the difference betwwen the unsupervised,
weakly-supervised and their combination, using a U-Net.

## Instructions

1.- Change the working directory to the root directory DeepReg.

2.- Run demo_data.py to download and extract all files, and to split the data into
training, validation and testing. If the data has already been downloaded. This will
also download the pre-trained models.

```
python ./demos/unpaired_ct_abdomen/demo_data.py
```

After running the command you will have the following directories in
DeepReg/demos/unpaired_ct_abdomen/dataset:

```
DeepReg$ ls demos/unpaired_ct_abdomen/dataset/
test  train  val
```

3.- The next step is to train the network using DeepReg. To train the network, run one
of the following commands in command line:

3.1- Unsupervised learning

```bash
deepreg_train --gpu "0" --config_path demos/unpaired_ct_abdomen/unpaired_ct_abdomen_unsup.yaml --log_dir unpaired_ct_abdomen_unsup
```

3.2- weakly-supervised learning

```bash
deepreg_train --gpu "1" --config_path demos/unpaired_ct_abdomen/unpaired_ct_abdomen_weakly.yaml --log_dir unpaired_ct_abdomen_weakly
```

3.3- Combined learning

```bash
deepreg_train --gpu "2" --config_path demos/unpaired_ct_abdomen/unpaired_ct_abdomen_comb.yaml --log_dir unpaired_ct_abdomen_comb
```

4.- After training the network, run demo_predict:

```
deepreg_predict --gpu "" --config_path demos/unpaired_ct_abdomen/unpaired_ct_abdomen_unsup.yaml --ckpt_path demos/unpaired_ct_abdomen/dataset/pre-trained/unsup/weights-epoch5000.ckpt --log_dir unpaired_ct_abdomen_unsup --save_png --mode test
```

```
deepreg_predict --gpu "" --config_path demos/unpaired_ct_abdomen/unpaired_ct_abdomen_weakly.yaml --ckpt_path demos/unpaired_ct_abdomen/dataset/pre-trained/weakly/weights-epoch5000.ckpt --log_dir unpaired_ct_abdomen_weakly --save_png --mode test
```

```
deepreg_predict --gpu "" --config_path demos/unpaired_ct_abdomen/unpaired_ct_abdomen_comb.yaml --ckpt_path demos/unpaired_ct_abdomen/dataset/pre-trained/comb/weights-epoch5000.ckpt --log_dir unpaired_ct_abdomen_comb --save_png --mode test
```

5.- Finally, if you want to see the result of predict with the images in the respective
test folders.

## Pre-trained Model

A pre-trained model is also available.

## Data

The data set is from the MICCAI Learn2Reg grand challenge
(https://learn2reg.grand-challenge.org/) task 3 [1], and can be downloaded directly from
https://learn2reg.grand-challenge.org/Datasets/

## Tested DeepReg Version

(tbc)

## References

[1] Adrian Dalca, Yipeng Hu, Tom Vercauteren, Mattias Heinrich, Lasse Hansen, Marc
Modat, Bob de Vos, Yiming Xiao, Hassan Rivaz, Matthieu Chabanas, Ingerid Reinertsen,
Bennett Landman, Jorge Cardoso, Bram van Ginneken, Alessa Hering, and Keelin Murphy.
(2020, March 19). Learn2Reg - The Challenge. Zenodo.
http://doi.org/10.5281/zenodo.3715652

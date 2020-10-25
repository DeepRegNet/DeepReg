# Unpaired abdomen CT registration

> **Note**: Please read the
> [DeepReg Demo Disclaimer](introduction.html#demo-disclaimer).

[Source Code](https://github.com/DeepRegNet/DeepReg/tree/main/demos/unpaired_ct_abdomen)

## Author

DeepReg Development Team (Ester Bonmati)

## Application

This demo shows how to register unpaired abdominal CT data from different patients using
DeepReg. In addition, the demo demonstrates the difference between the unsupervised,
weakly-supervised and their combination, using a U-Net.

## Data

The data set is from the MICCAI Learn2Reg grand challenge
(https://learn2reg.grand-challenge.org/) task 3 [1], and can be downloaded directly from
https://learn2reg.grand-challenge.org/Datasets/.

## Instruction

- [Install DeepReg](https://deepreg.readthedocs.io/en/latest/getting_started/install.html);

- Change the working directory to the root directory of DeepReg project;

- Run [demo_data.py] to download and extract all files, and to split the data into
  training, validation and testing. If the data has already been downloaded. This will
  also download the pre-trained models:

```bash
python demos/unpaired_ct_abdomen/demo_data.py
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
python demos/unpaired_ct_abdomen/demo_train.py --method unsup
```

- Weakly-supervised learning

```bash
python demos/unpaired_ct_abdomen/demo_train.py --method weakly
```

- Combined learning

```bash
python demos/unpaired_ct_abdomen/demo_train.py --method comb
```

- After training the network, run `demo_predict`:

The following example uses a pre-trained model, on CPU.

```bash
python demos/unpaired_ct_abdomen/demo_predict.py --method unsup
```

```bash
python demos/unpaired_ct_abdomen/demo_predict.py --method weakly
```

```bash
python demos/unpaired_ct_abdomen/demo_predict.py --method comb
```

- Finally, prediction results can be seen in the respective test folders in
  `logs_predict`.

## Pre-trained model

Three pre-trained models are available for this demo, for different training strategies
described above. These will be downloaded in respective sub-folders under the /dataset
folder using the `demo_data.py`. Run the user-trained model by specifying with
`--ckpt_path` the location where the checkpoint files are saved.

## Contact

Please [raise an issue](https://github.com/DeepRegNet/DeepReg/issues/new/choose).

## Reference

[1] Adrian Dalca, Yipeng Hu, Tom Vercauteren, Mattias Heinrich, Lasse Hansen, Marc
Modat, Bob de Vos, Yiming Xiao, Hassan Rivaz, Matthieu Chabanas, Ingerid Reinertsen,
Bennett Landman, Jorge Cardoso, Bram van Ginneken, Alessa Hering, and Keelin Murphy.
(2020, March 19). Learn2Reg - The Challenge. Zenodo.
http://doi.org/10.5281/zenodo.3715652

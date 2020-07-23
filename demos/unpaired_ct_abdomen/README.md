# Unpaired CT Abdominal Registration

## Author

Ester Bonmati (e.bonmati@ucl.ac.uk)

## Instructions

This demo shows how to register unpaired CT data from the abdomen using DeepReg (unlabeled and labeled).
The data that this demo uses is from the MICCAI Learn2Reg grand challenge (https://learn2reg.grand-challenge.org/) task 3 [1].

1.- First, download the data (L2R_Task3_AbdominalCT.tar) from Learn2Reg task3 to the following directory: demos/unpaired_ct_abdomen/dataset/. If you check the files in /demos/unpaired_ct_abdomen/dataset you should get the following:

```
DeepReg$ ls demos/unpaired_ct_abdomen/dataset/
L2R_Task3_AbdominalCT.tar
```

2.- Run demo_data.py to extract all files and to split the data in training, validation and testing. 

```
python ./demos/unpaired_ct_abdomen/demo_data.py  
```

After running the command you should see this:

```
DeepReg$ ls demos/unpaired_ct_abdomen/dataset/
L2R_Task3_AbdominalCT.tar  test  train  valid
```
Where L2R_Task3_AbdominalCT.tar is the original dataset file downloaded previously, test is a folder that contains the images and labels for testing, train is a folder that contains the images and labels for training, and valid is a folder that contains the images and labels for validation.

3.- The next step is to train the network using DeepReg. To train the network, run demo_train.py:

```
python ./demos/unpaired_ct_abdomen/demo_train.py   
```

4.- After training the network, run demo_predict:

```
python ./demos/unpaired_ct_abdomen/demo_predict.py   
```

5.- Finally, if you want to see the result of predict with the images in testing, run demo_plot:

```
python ./demos/unpaired_ct_abdomen/demo_plot.py   
```

## Application



## Data

The dataset for this demo is part of the Learn2Reg challenge (task 3) [1] and can be downloaded from:

https://learn2reg.grand-challenge.org/Datasets/


## Tested DeepReg Version

??

## References

[1] Adrian Dalca, Yipeng Hu, Tom Vercauteren, Mattias Heinrich, Lasse Hansen, Marc Modat, Bob de Vos, Yiming Xiao, Hassan Rivaz, Matthieu Chabanas, Ingerid Reinertsen, Bennett Landman, Jorge Cardoso, Bram van Ginneken, Alessa Hering, and Keelin Murphy. (2020, March 19). Learn2Reg - The Challenge. Zenodo. http://doi.org/10.5281/zenodo.3715652
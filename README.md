# DL_For_CV_Lightweight_Ensemble

This repository evalutes the effectiveness of an ensemble of 3 prominent lightweight models on 2 datasets, CIFAR-10 and AID. The code for downloading CIFAR-10 is already included in the respective scripts while the AID dataset can be downloaded from here: 

https://www.kaggle.com/datasets/jiayuanchengala/aid-scene-classification-datasets

To pretrain each of the models and generate the .pth files, please run the respective training scripts provided for each model provided with the AID dataset (if needed). After obtaining the .pth files for each model, please update the get_weight_paths function from the ensemble_model.py script before evaluating the ensemble. 
# Brats_2020_3DUnet [![Python 3.7](https://img.shields.io/badge/python-3.7-blue.svg)](https://www.python.org/downloads/release/python-360/) ![Python 3.7](https://img.shields.io/badge/Last%20Updated-30th%20Oct-green) 
## Description 
The assessment of cutting-edge techniques for the segmentation of brain tumors in multimodal magnetic resonance imaging (MRI) images has always been the main emphasis of BraTS. Pre-operative MRI images from many institutions are used in BraTS, which principally focuses on the segmentation (Task 1) of gliomas, which are fundamentally diverse (in appearance, form, and histology) brain tumors. Furthermore, BraTS concentrates on the prediction of patient overall survival in order to emphasize the clinical importance of this segmentation challenge (Task 2), As our final year project we performed Brain Tumor segmentation and survival prediction using BraTS20 dataset. In this approach we proposed the 3D U-Net model for Image segmentation and for survival prediction task autoencoder model is used.

## Table of Contents

1. [Installation](#Installation)
2. [Usage_Description](#Usage-Description)
3. [Comparision/Results](Comparisions/Results)
4. [Acknowledgements](#Acknowledgements)
5. [Support](#Support)
6. [License](#License)
7. [Project Status](#Project-Status)



## Installation
### Libraries setup

Install the libraries as follows
```bash
pip install requirements.txt
```
### Dataset:Google Drive Loading

Dataset can be downloaded from https://www.kaggle.com/datasets/awsaf49/brats20-dataset-training-validation

If using colab, after uploading mount the drive.

```bash
from google.colab import drive
drive.mount('./gdrive')
```
### Changing the global paths

Change the path to dataset and models in class GlobalConfig.

```bash
class GlobalConfig:
    root_dir = '/content/drive/MyDrive/final_dataset/FULL_BRATS_2020'
    train_root_dir = '/content/drive/MyDrive/final_dataset/FULL_BRATS_2020/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData'
    test_root_dir = '/content/drive/MyDrive/final_dataset/FULL_BRATS_2020/BraTS2020_ValidationData/MICCAI_BraTS2020_ValidationData'
    path_to_csv = '/content/drive/MyDrive/final_dataset/train_data.csv'
```

## Usage-Description
### Dataset

All BraTS multimodal scans are available as NIfTI files (.nii.gz) and describe native (T1) and post-contrast T1-weighted (T1Gd), T2-weighted (T2), and d) T2-FLAIR volumes. They were acquired using various clinical protocols and scanners from multiple (n=19) institutions, which are mentioned as data contributors. All the imaging datasets have been segmented manually, by one to four raters, following the same annotation protocol, and their annotations were approved by experienced neuro-radiologists. Annotations comprise the GD-enhancing tumor (ET — label 4), the peritumoral edema (ED — label 2), and the necrotic and non-enhancing tumor core (NCR/NET — label 1). This dataset is made up of 3D MRI brain scans from 369 individuals with gliomas, 76 of them have LGG, and the remaining patients have HGG.

### Task 1


### Task 2

## Comparisions/Results

## Acknowledgements
We would like to express our gratitude to Dr. S. Mohammed to help us in our research and providing us valuable pointers for out project. We would also like to express our thanks to Peralman School of medicine, University of Pennysylvania for providing providing us the data needed for this project. Lastly we would also like express our gratitude to Lakehead University for facilliating this project as our final year project.

## License
This is open source project. Though it would be nice to give us message if find this project useful for your needs.

## Support
Feel free to contact me in case code has bugs. In fact I will very much appreciate for finding the faults in the code. Feel free to reach out to me. Though emails are the fastest ones I reply.

Connect: Yash Atul Patel ![twitter](https://img.shields.io/twitter/follow/yashpatel?style=social)![social](https://img.shields.io/github/followers/YashPatel91?style=social) 
Contact: yash9132@gmail.com 

Connect: KrushangBhavsar ![social](https://img.shields.io/github/followers/KrushangBhavsar?style=social) 
Contact: krushangbhavsar@gmail.com


Connect: Abhishek Shah ![social](https://img.shields.io/github/followers/shah1411?style=social) 
Contact: abhishekshah1411@gmail.com

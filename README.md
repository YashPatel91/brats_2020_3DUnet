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

### Task 1


### Task 2

## Comparisions/Results

As it stands now, [Model-1](#Model-1-Transfer-learning-Inception-V3) has accuracy of around 82% accuracy over the test data set while the custom [Model-2](#Model-2-Custom-CNN-model_1) has accuracy of around 70% at the end of its run. [Model-3](#Model-3-Custom-CNN-model_2) is highly process expensive, a single epoch has processing time of around 150 minutes. [Model-3](#Model-3-Custom-CNN-model_2) has achived accuracy over 85% over 7 epochs.

Computationally [Model-1](#Model-1-Transfer-learning-Inception-V3) has the highest efficiency for the identification to processing time ratio while [Model-3](#Model-3-Custom-CNN-model_2) has the highest accuracy but at the cost very processing needs.


## Road-map

Currently the classification is done with straigh forward approach of extracting features and comparing them with the trained weights. Here images are directly used by the model to predict their class without any other form of manipulation of the data before and after the model. So I propose use of object identifier like Yolo or denseNet to first identify a mushroom from the picture and make a bounding box around it. After this, the image in the bounding box will be used to train or predict in the model. This approach I belive would be better suited for the needs of this project

Additionally this classification is done considering only 4 mushrooms as its dataset. Out of these 2 are edible and 2 are poisonous to have uniform data. Currently these mushrooms constitute major part of all mushrooms used worldwide. I think model can be further improved with the help wider range of dataset. This will allow greater distinction between poisonous and edible mushrooms.

## Acknowledgements
I would like to express my gratitude to Lakehead University for providing me with an opportunity to do this research. I would also like to express my thanks to Dr. Trevor Tomesh for guiding me through my journy of writing this research paper and help me get better understanding of github and its uses.

## License
This is open source project. Though it would be nice to give me message if find this project useful for your needs.

## Support
Feel free to contact me in case code has bugs. In fact I will very much appreciate for finding the faults in the code. Feel free to reach out to me. Though emails are the fastest ones I reply.



## Project-Status
Project is still under development but on hold for indefinate amount. Would be great if anyone wants to collaborate for this. Just drop a message

Connect: Yash Atul Patel ![twitter](https://img.shields.io/twitter/follow/yashpatel?style=social)![social](https://img.shields.io/github/followers/YashPatel91?style=social) 

Contact: yash9132@gmail.com

# Brats_2020_3DUnet [![Python 3.7](https://img.shields.io/badge/python-3.7-blue.svg)](https://www.python.org/downloads/release/python-360/) ![Python 3.7](https://img.shields.io/badge/Last%20Updated-30th%20Oct-green) ![Python 3.7](https://img.shields.io/github/contributors/yashpatel91/poisonous_mushroom_classification)
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
### Preprocessing for training without image generator

Without image generator, code and preprocessing gets simpler. You won't need to manage the images,labels and other markers for the training in proper order. A simple loading can be done as follows
```bash
labels = ["Agaricus Bisporus", "Death Cap", "Galerina Autumnalis", "Oyster"]

for i,label in enumerate(labels):
  path = os.path.join(os.getcwd(), label)
  images = os.listdir(path)
  for img in images:
    image_loc = os.path.join(path, img)
    try:
      ip = cv.imread(image_loc)
      ip = cv.resize(ip, (300,300))
      data.append([ip, i])
    except:
      print("Image read error at", label, img)
```
After this data can be pickled for further use or just shuffled for training
```bash
random.shuffle(data)
len(data)
```


### Preprocessing for training with Data generator

Without data generator training would have limitation with regards to the amount of ram that can be used. This is where data generator comes in place. It feeds the training model at runtime to unable ram clogging with huge data set and model. This approach is a general way of preprocessing even in the current industry cause most the the industry level projects have dataset in millions that won't be feasible to allocate memory to in single go. As such data generators are used. They can also be used to transform data such that a single data entity can generate multiple data entity with different transformations. Here I haven't used any kind of transformation for the dataset to be used.

```
def data_generator(samples, batch_size=32,shuffle_data=True,resize=224):
    """
    Yields the next training batch.
    Suppose `samples` is an array [[image1_filename,label1], [image2_filename,label2],...].
    """
    num_samples = len(samples)
    while True: # Loop forever so the generator never terminates
        samples = shuffle(samples)

        # Get index to start each batch: [0, batch_size, 2*batch_size, ..., max multiple of batch_size <= num_samples]
        for offset in range(0, num_samples, batch_size):
            # Get the samples you'll use in this batch
            batch_samples = samples[offset:offset+batch_size]

            # Initialise X_train and y_train arrays for this batch
            X_train = []
            y_train = []

            # For each example
            for batch_sample in batch_samples:
                # Load image (X) and label (y)
                img_name = batch_sample[0]
                label = batch_sample[1]
                img =  cv2.imread(os.path.join("",img_name))
                
                # apply any kind of preprocessing
                img,label = preprocessing(img,label)
                # Add example to arrays
                X_train.append(img)
                y_train.append(label)

            # Make sure they're numpy arrays (as opposed to lists)
            X_train = np.array(X_train)
            y_train = np.array(y_train)

            # The generator-y part: yield the next training batch            
            yield X_train, y_train
```

To get the data in ready formate for our data generator I've made a script for that specific purpose seperatly which is __make_generator_dataset.ipynb__. Here this pre processing data set algorithm is not specific to this classification. 

```
num_classes = 4
labels_name={'Agaricus Bisporus':0,'Death Cap':1,'Galerina Autumnalis':2,'Oyster':3}
num_images_for_test = 200
train_df = pd.DataFrame(columns=['FileName', 'Label', 'ClassName'])
test_df = pd.DataFrame(columns=['FileName', 'Label', 'ClassName'])
data_dir_list = ["Agaricus Bisporus", "Death Cap", "Galerina Autumnalis", "Oyster"]
```
It can easily be altered to suite other classifications. With following parameters to be taken into consideration

```
num_classes = 4
```
Represents the number classes to be classified.

```
labels_name={'Agaricus Bisporus':0,'Death Cap':1,'Galerina Autumnalis':2,'Oyster':3}
```
Represents the dictionionary for labels to classes.

```
train_df = pd.DataFrame(columns=['FileName', 'Label', 'ClassName'])
test_df = pd.DataFrame(columns=['FileName', 'Label', 'ClassName'])
```
Represents the formating of the csv file for its use by datagenerator for ease of loading images at runtime

### Model-1-Transfer-learning-Inception-V3

This model is present in __training_without_data_generator_transfered_learning.ipynb__. This model employs transfer learning by employing Inception V3 in its architecture to do its prediction. As this model is not too complex and has lower number of parameters, it doesn't require data generator to train on the data-set. It directly trains over whole data-set in a single go. Below is the model architecture.

```
mdl1 = Sequential()
mdl1.add(InceptionV3(input_shape=(300,300,3), include_top=False, weights='imagenet'))
mdl1.add(Flatten())
mdl1.add(Dense(64, activation='relu'))
mdl1.add(Dense(4, activation='sigmoid'))
mdl1.compile(loss='categorical_crossentropy', metrics=['acc'], optimizer='sgd')
```
Its training graphs are as follows:

![python](readme_images/inception_graph.PNG)

### Model-2-Custom-CNN-model_1

This is a convolutional neural model which doesn't have transfered learning weights as the previous approach. This approach was designed to better understand in which direction to make the further changes to convolutional neural network such that it can do better at classifing mushrooms. This approach seems to be inferior to the transfer learning approach but it be used to learn to make a better model.Below is its architecture.

```
model.add(Conv2D(32, (3,3),padding='same',input_shape=input_shape,name='conv2d_1'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2),name='maxpool2d_1'))
model.add(Conv2D(32, (3, 3),name='conv2d_2'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2),name='maxpool2d_2'))
model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(4))
model.add(Activation('softmax'))
```
Its training graphs are as follows:

![python](readme_images/test_graphs.png)

### Model-3-Custom-CNN-model_2

This model is based on testing of the previous approach. Though this model is computationally expensive its able to get good accuracy right from first epoch. This architecture is a combination of Sequential(), Conv2D(), Batch Normalization, Max Pooling, Dropout, and Flatting. Its architecture is as follows.

```

model = Sequential()
model.add(Conv2D(filters=64, kernel_size=(5, 5), input_shape=input_shape, activation='relu'))
model.add(BatchNormalization(axis=3))
model.add(Conv2D(filters=64, kernel_size=(5, 5), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(BatchNormalization(axis=3))
model.add(Dropout(0.1))
model.add(Conv2D(filters=128, kernel_size=(5, 5), activation='relu'))
model.add(BatchNormalization(axis=3))
model.add(Conv2D(filters=128, kernel_size=(5, 5), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(BatchNormalization(axis=3))
model.add(Dropout(0.1))
model.add(Conv2D(filters=256, kernel_size=(5, 5), activation='relu'))
model.add(BatchNormalization(axis=3))
model.add(Conv2D(filters=256, kernel_size=(5, 5), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(BatchNormalization(axis=3))
model.add(Dropout(0.1))
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Dense(256, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Dense(4, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()
```
Due to its number of parameters it takes more then 2 hours in google collabe to run its first epoch

![python](readme_images/cnn_arch.PNG)

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

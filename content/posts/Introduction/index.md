---
title: "Depth estimation project review"
date: 2022-06-17T08:06:25+06:00
description: Introduction to Sample Post
menu:
  sidebar:
    name: Introduction
    identifier: introduction
    weight: 10
tags: ["Basic", "Multi-lingual"]
categories: ["Basic"]
---

### Introduction

Today we're going to take a look at a computer vision project on depth estimation and see how we can improve it.
The goals of this investigation are twofold

- Highlight the good and bad practices used in the project
- Apply MLOps techniques to serve a production model in the cloud

A few of the techniques to be included are the following: Unit testing, CI/CD, cloud serving, monitoring, and model experimentation and evaluation.
If you wish to read the original Jupyter notebook from the project then feel free to read it here. (Link to be added)

### Project proposal

The goal of the original project was to create a dataset of RGBA and depth image pairs from the popular video game Grand Theft Auto V, GTAV. This dataset would then be used to train a convolutional neural network to predict a depth image from an RGBA image. Using the trained model, a dataset of real life images was used to evaluate the model's ability to predict real world depth after training on this synthetic video game data. I will upload the entire project proposal document for you to read at your leisure, but this is the gist of the project.


{{< img src="images/colour_simple.png" align="center" >}} 
*Figure 1: Example colour image from GTAV within the training dataset. A little close for liking...*
{{< vs 3 >}}

{{< img src="images/depth_simple.png" align="center" >}} 
*Figure 2: Associated depth image from the same dataset.*
{{< vs 3 >}}


### Approach


I will read through the project as it currently stands and will critique the project organisation and structure. I'll then follow the data from collection to pre-processing to get a feel for the data pipeline, then look at how the model architectures were defined. After this I'll see how training, validation and testing loops worked and give them a good talking to. I'll see what analysis techniques were used to analyse the predictions from the trained models and how these were interpreted. After this analysis, I will propose a series of improvements that will be made to take the project to a production standard.


### Project structure

Put simply, the project structure is poor. This is somewhat to be expected from a group of three masters students working together on their own jupyter notebooks and computers. Examples of bad practices found are:
 - Inconsistent folder and file naming conventions (or none at all)
 - Monolithic jupyter notebooks
 - Attempt at containerisation but not fully working
 - No full list of package requirements
 - Different flietypes stored in same folders (models stored together with README and datasets)
 - Just check out the structure below...

{{< img src="images/project structure.png" align="center" >}} 
*Figure 3: Typical organisation structure of a group of researchers working together on a data science project.*
{{< vs 3 >}}


### Data acquisition

A synthetic dataset was collected from GTAV in three stages: Simple collection, Moderate collection, and Hard collection.
These are defined below:

{{< img src="images/collection_definitions.png" align="center" >}} 
*Figure 4: Defining the data collection methods used to create three datasets for training, validating, and testing.*
{{< vs 3 >}}

Data for the project was stored on Microsoft's OneDrive in a zipped folder that needed to be downloaded and extracted before it could be used by another researcher.
Data was shared between the group members during experimentation and had to be loaded onto a local machine in order to train a model.
During the project, only the Simple and moderate datasets were collected. The moderate dataset is outlined below:

{{< img src="images/moderate_data.png" align="center" >}} 
*Figure 5: Moderate dataset overview collected from GTAV.*
{{< vs 3 >}}

### Data preprocessing

Since the Simple collection dataset is only small, we will focus on the work done with the Moderate dataset.
Structuring of this dataset is defined in a .csv file that lists the splitting of data by weather conditions.
A python dataset class was created to read the data from a local machine to a pytorch dataset. Unfortunately, the method of reading data to this class is inefficient and clumsy. It relies on the use of "f-strings" to read in files from different folders, many "if-else" statements, and hardcoded paths. There is little doccumentation of the code and no unit testing for individual components of the pipeline. Although this code is not clean, it works for the specific use case it was written for, which tends to be the case in research environments.

The Moderate dataset class can be seen below, beware of some codesmells...

``` python
class ModerateDataset(Dataset):

    def __init__(self, col_dir='', depth_dir='', transform=None, trans_on=False):
        self.path_names = {}
        for folder in folder_names:
            self.path_names[f"{folder}"] = {}
        for folder in folder_names:
            self.path_names[f'{folder}']['colour'] = {}
            self.path_names[f'{folder}']['depth'] = {}
        for i in range(1, num_files[0]):
            self.path_names['Sunny']['colour'][f"{i}"] = {}
            self.path_names['Sunny']['depth'][f"{i}"] = {}
        print("*************MAKE SURE THE PATH FILE IN THE FOR LOOP IS THE BASE IMAGE DIRECTORY ON YOUR COMPUTER**************")
        count = 0
        for folder in folder_names:
            for i in range(0, num_files[folder_names.index(folder)]):
                self.path_names[f'{folder}']['colour'][f'{i+1}'] = Path(f"C:/Users/Admin/OneDrive/Computer Vision/Moderate collection/{folder}/colour/{colour_filenames[count+i]}")  ## Change this path here!!!!
                self.path_names[f'{folder}']['depth'][f'{i+1}'] = Path(f"C:/Users/Admin/OneDrive/Computer Vision/Moderate collection/{folder}/depth/{depth_filenames[count+i]}")   ## Change this path here!!!!
            count = count + num_files[folder_names.index(folder)]
        
        self.transform = transform
        self.col_dir = col_dir
        self.depth_dir = depth_dir
        self.trans_on = trans_on

    def __getitem__(self,idx):
        if idx == 0:
            
            self.col_dir = self.path_names[f'{folder_names[0]}']['colour'][f'{idx+1}']
            self.depth_dir = self.path_names[f'{folder_names[0]}']['depth'][f'{idx+1}']
        
        if (idx>0 and idx <= num_files[0]):  ## 1-500

            self.col_dir = self.path_names[f'{folder_names[0]}']['colour'][f'{idx}']
            self.depth_dir = self.path_names[f'{folder_names[0]}']['depth'][f'{idx}']

        elif (idx > num_files[0] and idx < (sum(num_files[:2])+1)): ## 501 - 1500

            self.col_dir = self.path_names[f'{folder_names[1]}']['colour'][f'{idx-num_files[0]}']
            self.depth_dir = self.path_names[f'{folder_names[1]}']['depth'][f'{idx-num_files[0]}']

        elif (idx > sum(num_files[:2]) and idx < (sum(num_files[:3])+1) ): ## 1501 - 2600

            self.col_dir = self.path_names[f'{folder_names[2]}']['colour'][f'{idx-sum(num_files[:2])}'] # -1500
            self.depth_dir = self.path_names[f'{folder_names[2]}']['depth'][f'{idx-sum(num_files[:2])}']

        elif (idx > sum(num_files[:3]) and idx < (sum(num_files[:4])+1) ): ## 2601 - 5600

            self.col_dir = self.path_names[f'{folder_names[3]}']['colour'][f'{idx-sum(num_files[:3])}'] #-2600
            self.depth_dir = self.path_names[f'{folder_names[3]}']['depth'][f'{idx-sum(num_files[:3])}']
            
        elif (idx > sum(num_files[:4]) and idx < (sum(num_files[:5])+1) ): ## 5601 - 7857

            self.col_dir = self.path_names[f'{folder_names[4]}']['colour'][f'{idx-sum(num_files[:4])}'] # -5600
            self.depth_dir = self.path_names[f'{folder_names[4]}']['depth'][f'{idx-sum(num_files[:4])}']

        elif (idx > sum(num_files)):
            raise NameError('Index outside of range')

        col_img = import_raw_colour_image(self.col_dir)
        depth_img = import_raw_depth_image(self.depth_dir)
        if self.trans_on == True:
            col_img = torch.from_numpy(np.flip(col_img,axis=0).copy()) # apply any transforms
            depth_img = torch.from_numpy(np.flip(depth_img,axis=0).copy()) # apply any transforms
            col_img = col_img.transpose(0,2)
            col_img = col_img.transpose(1,2)
        if self.transform: # if any transforms were given to initialiser
            col_img = self.transform(col_img) # apply any transforms
        return col_img, depth_img
    
    def __len__(self):
        return sum(num_files)
```

Creating an instance of this dataset class into the variable ```total_data``` allowed the splitting of data into three components.

``` python
total_data = ModerateDataset(trans_on=True) ## Instantiating the dataset
```

It was of importance to establish the separation of three datasets: training, validation and testing. Training data was used to train the neural network model and validation data is used to check that the model was not overfitting to the training data. Testing data was used to check the performance of the trained model on unseen data to evaluate performance with a set of predefined metrics, defined in a later section.

A train, validation, testing split of 80/10/10 has been used to create three datasets: ```train_dataset```, ```val_dataset``` and ```test_dataset```. This split is commonly used in machine learning research. These datasets all inherit from the ```ModerateDasaset``` class.

``` python
train_size = int(0.8 * len(total_Data)) # Size of training dataset (80% of total)
val_size = int((len(total_Data) - train_size)/2) ## Size of validation and test datasets (10% of total)
train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(total_Data, [train_size, val_size, val_size]) # train, val, and test splits
```


 For each of these datasets, a data loader was created to load in a batch of images at once instead of loading the entire dataset to memory. To train the model, the training and validation dataloaders are used. This ensures that no testing data is used in any step of training the model.

``` python
batch_sz = 16 # Batch size
tr_dl  = DataLoader(train_dataset,  batch_size=batch_sz, shuffle=True,  num_workers=0) # Training dataloader
val_dl = DataLoader(val_dataset,  batch_size=batch_sz, shuffle=True,  num_workers=0)   # Validation dataloader
test_dl = DataLoader(test_dataset,  batch_size=batch_sz, shuffle=True,  num_workers=0) # Test dataloader
```
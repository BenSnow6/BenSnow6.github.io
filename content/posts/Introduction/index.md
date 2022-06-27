---
title: "Depth estimation project review"
date: 2022-06-27T08:06:25+06:00
description: Depth Estimation project in review
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
If you wish to read the original Jupyter notebook from the project then feel free to read it here. (Link to be added after anonymity is included)

### Project proposal

The goal of the original project was to create a dataset of RGB and depth image pairs from the popular video game Grand Theft Auto V, GTAV. This dataset would then be used to train a convolutional neural network to predict a depth image from an RGB image. Using the trained model, a dataset of real-life images was used to evaluate the model's ability to predict real-world depth after training on this synthetic video game data. I will upload the entire project proposal document for you to read at your leisure, but this is the gist of the project.


{{< img src="images/colour_simple.png" align="center" >}} 
*Figure 1: Example colour image from GTAV within the training dataset. A little close for liking...*
{{< vs 3 >}}

{{< img src="images/depth_simple.png" align="center" >}} 
*Figure 2: Associated depth image from the same dataset.*
{{< vs 3 >}}


### Approach


I will read through the project as it currently stands and will critique the project's organisation and structure. I'll then follow the data from collection to pre-processing to get a feel for the data pipeline, then look at how the model architectures were defined. After this, I'll see how training, validation, and testing loops worked and give them a good talking to. I'll see what analysis techniques were used to analyse the predictions from the trained models and how these were interpreted. After this analysis, I will propose a series of improvements that will be made to take the project to a production standard.


### Project structure

Put simply, the project structure is poor. This is somewhat to be expected from a group of three master's students working together on their own jupyter notebooks and computers. Examples of bad practices found are:
 - Inconsistent folder and file naming conventions (or none at all)
 - Monolithic jupyter notebooks
 - Attempt at containerisation but not fully working
 - No full list of package requirements
 - Different filetypes stored in the same folders (models stored together with README and datasets)
 - Just check out the structure below...

{{< img src="images/project structure.png" align="center" >}} 
*Figure 3: Typical organisation structure of a group of researchers working together on a data science project.*
{{< vs 2 >}}


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

Points to improve:
- Data collected is inaccessible and inflexible
- Data is not versioned
- Metadata saved in external .csv file

### Data preprocessing

Since the Simple collection dataset is only small, we will focus on the work done with the Moderate dataset.
The structuring of this dataset is defined in a .csv file that lists the splitting of data by weather conditions.
A python dataset class was created to read the data from a local machine to a pytorch dataset. Unfortunately, the method of reading data in this class is inefficient and clumsy. It relies on the use of "f-strings" to read in files from different folders, many "if-else" statements, and hardcoded paths. There is little documentation  of the code and no unit testing for individual components of the pipeline. Although this code is not clean, it works for the specific use case it was written for, which tends to be the case in research environments.

Some data preprocessing was performed on the dataset including some transposes and conversions to torch tensors from numpy arrays.

The Moderate dataset class can be seen below, beware of some code smells...

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

It was of importance to establish the separation of three datasets: training, validation, and testing. Training data was used to train the neural network model and validation data is used to check that the model was not overfit to the training data. Testing data was used to check the performance of the trained model on unseen data to evaluate performance with a set of predefined metrics, defined in a later section.

A train, validation and, testing split of 80/10/10 has been used to create three datasets: ```train_dataset```, ```val_dataset``` and ```test_dataset```. These datasets all inherit from the ```ModerateDasaset``` class.

``` python
train_size = int(0.8 * len(total_Data)) # Size of training dataset (80% of total)
val_size = int((len(total_Data) - train_size)/2) ## Size of validation and test datasets (10% of total)
train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(total_Data, [train_size, val_size, val_size]) # train, val, and test splits
```


 For each of these datasets, a data loader was created to load a batch of images at once instead of loading the entire dataset to memory. To train the model, the training and validation dataloaders are used. This ensures that no testing data is used in any step of training the model.

``` python
batch_sz = 16 # Batch size
tr_dl  = DataLoader(train_dataset,  batch_size=batch_sz, shuffle=True,  num_workers=0) # Training dataloader
val_dl = DataLoader(val_dataset,  batch_size=batch_sz, shuffle=True,  num_workers=0)   # Validation dataloader
test_dl = DataLoader(test_dataset,  batch_size=batch_sz, shuffle=True,  num_workers=0) # Test dataloader
```

Points to improve
- Inflexible dataset reading, can't add new data easily
- None of the code is unit tested
- Inconsistent naming conventions
- Hard coded values and filepaths
- If-Else statements aplenty

### Defining the model

A simple CNN architecture was developed as a baseline model to compare performance against. Ideally, the team planned on using a more complex model to experiment with but time constraints made this unfeasable. The neural network is defined with pytorch to be the following:

``` python
net = nn.Sequential(
    nn.Conv2d(in_channels=3,  out_channels=6, kernel_size=3, stride=1, padding=1), 
    nn.ReLU(),
    nn.Conv2d(in_channels=6, out_channels=12, kernel_size=3, stride=1, padding=1),
    nn.ReLU(),
    nn.ConvTranspose2d(in_channels = 12, out_channels=6, kernel_size=3, stride=1, padding=1),
    nn.ReLU(),
    nn.ConvTranspose2d(in_channels = 6, out_channels=1, kernel_size=3, stride=1, padding=1),
    nn.Sigmoid()
).cuda()
```

Using model summary, we can see the network's structure when given an input of shape (3,720,1280), this being the resolution of the collected colour images.

{{< img src="images/simpleCNN.png" align="center" >}} 
*Figure 6: A simple CNN architecture used to convert colour images to depth images.*
{{< vs 3 >}}

After creating the pytorch model, a training loop was developed to train the network on the training dataset whilst validating with the validation dataset. A description of the hyperparameters used to train the model can be seen below.

{{< img src="images/simplecnn_desc.png" align="center" >}} 
*Figure 7: Simple CNN hyperparameters and tracked variables during training.*
{{< vs 3 >}}

Points to improve
- Hyperparameter search
- Experiment tracking
- Different model architecture search

The training loop itself can be seen below.

``` python
def fit(net, tr_dl, val_dl, loss=nn.MSELoss(), epochs=3, lr=3e-3, wd=1e-3):   

    Ltr_hist, Lval_hist = [], []    
    opt = optim.Adam(net.parameters(), lr=lr, weight_decay=wd)
    for epoch in trange(epochs):
        
        L = []
        dl = (iter(tr_dl))
        count_train = 0
        for xb, yb in tqdm(dl, leave=False):
            xb, yb = xb.float(), yb.float()
            xb, yb = xb.cuda(), yb.cuda()
            y_ = net(xb)
            l = loss(y_, yb)
            opt.zero_grad()
            l.backward()
            opt.step()
            L.append(l.detach().cpu().numpy())
            print(f"Training on batch {count_train} of {int(train_size/batch_sz)}")
            count_train+= 1

        # disable gradient calculations for validation
        for p in net.parameters(): p.requires_grad = False

        Lval, Aval = [], []
        val_it = iter(val_dl)
        val_count = 0
        for xb, yb in tqdm(val_it, leave=False):
            xb, yb = xb.float(), yb.float()
            xb, yb = xb.cuda(), yb.cuda()
            y_ = net(xb)
            l = loss(y_, yb)
            Lval.append(l.detach().cpu().numpy())
            Aval.append((y_.max(dim=1)[1] == yb).float().mean().cpu().numpy())
            print(f"Validating on batch {val_count} of {int(val_size/batch_sz)}")
            val_count+= 1

        # enable gradient calculations for next epoch 
        for p in net.parameters(): p.requires_grad = True 
            
        Ltr_hist.append(np.mean(L))
        Lval_hist.append(np.mean(Lval))
        print(f'training loss: {np.mean(L):0.4f}\tvalidation loss: {np.mean(Lval):0.4f}\tvalidation accuracy: {np.mean(Aval):0.2f}')
    return Ltr_hist, Lval_hist
```

Just like the code seen before, there are quite a few improvements that can be made to bring the quality to a higher standard. Some of these include:

- Remove timing iterators
- Naming conventions
- Remove print functions
- Tracking/ logging during training
- Model versioning

### Model evaluation

A trained SimpleCNN model was saved so that it could be reused for the evaluation stage at a later date. This worked for the project, but a more rigorous approach can be used in future. Linking the model to the data used to train it and the hyperparameters used during training for example. Along with this, storing the model in a repository so that other users can easily access it and understand where it came from and how it was trained.

During the evaluation stage, the validation dataset was used to predict depth images from the RGB images. A series of metrics were then calculated to compare the predicted depth images and the ground truth data. This process was then repeated for the test set data that the model was not trained with. In total, nine metrics were calculated for all 786 test images. For this testing dataset, the mean and standard deviation of these metrics were calculated. A list of these metrics for the validation and test sets can be seen below.

{{< img src="images/val_test_errors.png" align="center" >}} 
*Figure 8: Calculated metrics with their mean and standard deviations for the validation and test datasets.*
{{< vs 3 >}}


After evaluation on the test set, some discussion of the metrics was made to understand the predictions made by the model. It would have been nice to see some more visual analysis, perhaps an ablation study to assess the features in different layers of the CNN. It would have also been nice to see some side-by-side comparisons of the depth image predictions and ground truths to look for artefacts where the model is lacking in accuracy. Since metrics were calculated in a pixel-wise fashion, it would be interesting to see them plotted as an image where each pixel represents the metric calculated at that point in the image. This would help give a visual representation of where in the image the error metrics are high and low along with any other high level features that may be occuring.

An external dataset, called the KITTI dataset, was also used for evaluation. A subset of the KITTI dataset used for evaluation consisted of 1000 RGB, depth image pairs taken from Lidar sensors attatched to a car that was driving down a suburban road.

{{< img src="images/kittiRGB.png" align="center" >}} 
*Figure 9: Sample RGB image from the KITTI dataset.*
{{< vs 3 >}}

{{< img src="images/kittidepth.png" align="center" >}} 
*Figure 10: Associated depth image from KITTI dataset.*
{{< vs 3 >}}

{{< img src="images/simplekittidepth.png" align="center" >}} 
*Figure 11: Predicted depth image from SimpleCNN model.*
{{< vs 3 >}}

Additional explanation of the depth images generated from predictions of the SimpleCNN model would be nice. It would be interesting to explain the colours used in the plots and the min/max values of each depth image to better understand what has been calculated. A list of relationships required to do this are highlighted below.

{{< img src="images/requiredData.png" align="center" >}} 
*Figure 12: Additional relationships required to progress the evaluation of the SimpleCNN's predictions against the KITTI benchmark.*
{{< vs 3 >}}

Error metrics were calculated with a mean and standard deviation allowing paving way for hypothesis testing upon changing model hyperparameters. Unfortunately, the error metrics calculated are missing some scaling factors to allow for direct comparison between the Moderate dataset depth predictions and the KITTI dataset depth. If these factors are found, this will allow a swath of analysis to be conducted. Conversions from the current error values to SI units would be a great improvement.

Error analysis can be improved by assessing the quality of the metrics themselves and interpreting their meaning in relation to the task. During my physics degree, I enjoyed this aspect of analysis. As such, in this revamp of the project I'm keen to look more in depth at the error analysis and metrics.

Points to improve
- Model versioning
- Metric evaluation
- Error analysis
- Depth conversion and comparison

### Summary

Having looked through the project, there are many good techniques used and a decent overarching methodology presented. To take the project to another level it will be useful to flesh out the experimentation procedure and increase the reusability and scalability of the code used. Best practices involving unit testing, code versioning, continuous integration/ continuous deployment, and code documentation are to be used. Utilising these best practices will make the project more robust and scalable. Read the next post to see what the first steps of this process are.
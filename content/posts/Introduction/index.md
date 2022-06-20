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

### Project proposal

The goal of the original project was to create a dataset of RGBA and depth image pairs from the popular video game Grand Theft Auto V. This dataset would then be used to train a convolutional neural network to predict a depth image from an RGBA image. Using the trained model, a dataset of real life images was used to evaluate the model's ability to predict real world depth after training on this synthetic video game data. I will upload the entire project proposal document for you to read at your leisure, but this is the gist of the project.


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
 - Attempt at contanerisation but not fully working
 - Different flietypes stored in same folders (models stored together with README and datasets)
 - Just check out the structure below...

{{< img src="images/project structure.png" align="center" >}} 
*Figure 3: Typical organisation structure of a group of researchers working together on a data science project.*
{{< vs 3 >}}

---
title: "Restructuring a project"
date: 2022-06-27T08:06:25+06:00
description: Taking a current, unorganized project and restructuring it for more efficient development.
menu:
  sidebar:
    name: Restructuring
    identifier: Restructuring
    weight: 10
tags: ["Basic", "Multi-lingual"]
categories: ["Basic"]
---

### Introduction

Jumping right in, let's start by creating a new folder on the desktop, a new conda environment, and activate that environment.

```bash
mkdir ~/Desktop/DepthEstimation
conda create --name depth_env
conda activate depth_env
```
I will be using the CookieCutter Data Science template to create a new project structure. This is done with the following commands:
    
```bash
pip install cookiecutter
cookiecutter -c v1 https://github.com/drivendata/cookiecutter-data-science
```

After filling in the CookieCutter template, we can navigate to the newly created project and start working on it. It should look something like this:

{{< img src="images/CookieCutter.png" align="center" >}} 
*Figure 1: CookieCutter project structure*
{{< vs 3 >}}

Installing the required dependencies is as simple as running the following command:

```bash
pip install -r requirements.txt
```

After installing the dependencies, we push the repository to github on the main branch which can be seen at the following address; https://github.com/BenSnow6/DepthEstimation. Instead of working on this main branch, I will create a new branch, develop a feature on this branch, and then merge the with a pull request to the main branch. This will ensure that the main branch is never directly edited and is only updated when the CI pipeline tests pass. This will be covered in a later section. For now, let's create a new branch called 'NotebookExploration' and open that in our IDE.

After looking through the previous post and the notebook, I have reduced the number of cells down to just the ones I need to work on. From this, I will split the cells into separate python files in the folder structure I created earlier. Splitting the notebook into python files is a good idea because it will make it easier to work on individual aspects of the project without being overfaced with the complexity of a single notebook. Along with splitting the code into separate files, I will perform unit tests on the code to ensure that it is working as intended.


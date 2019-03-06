---
layout: post
title: "Classifying Fe-Oxidizing Microbial Colonies"
date: 2018-07-20
excerpt: "Custom convolutional neural network from scratch using Keras."
tags: [machine-learning, oceanography]
comments: false
project: true
feature: "../assets/post/deep-sea/colony0_1024.png"
---

All real science, publications, and amazing work done on the expedition (not done by me) can be found <a href="https://earthref.org/FEMO/loihi.htm"><b>here</b></a> and on <a href="https://www.soest.hawaii.edu/oceanography/glazer/Brian_T._Glazer/Publications/Publications.html"><b>Brian Glazer's website</b></a>  I was only fortunate enough to have the chance to see the data and make a project of it. 


The hostile environment of an active submarine volcano being a candidate suitable for life to exist is shocking. Also quite humbling. The University of Hawaii does a tremendous amount of research at the Loihi Seamount near the Big Island, I was lucky enough to get a dataset of ~30,000 images from an expedition thanks to Dr. Brian Glazer. Given the size of the dataset and the nature of the expedition, most of the images are either a 'blank' seafloor (see pic below) or a select few of interest that have Fe-oxidizing microbial (FeM) colonies. The task of a classifier would be to sift the large dataset for only the ones that have interesting surface features and FeM.

## The Dataset:
The first dataset I gathered was relatively straightforward. It was a manner of "Colony present" or "None". An example of the two classes are below with 100 images each.. 

<figure class="half">
    <a href="../assets/post/deep-sea/colony0_1024.png"><img src="../assets/post/deep-sea/colony0_1024.png"></a>
    <a href="../assets/post/deep-sea/none0_1024.png"><img src="../assets/post/deep-sea/none8_1024.png"></a>
    <figcaption>Class of interest vs 'blank' seafloor.</figcaption>
</figure>

## Model Architecture v1:
The first model was straightforward, three convolutional layers with a final fully connected layer on top.  

<figure>
    <a href="../assets/post/deep-sea/network.png"><img src="../assets/post/deep-sea/architecture.png"></a>
    <figcaption>Class of interest vs 'blank' seafloor.</figcaption>
</figure>


Starting simple gave a good chance to understand the initial sense of workflow & what deep-sea features would be of interest. Perhaps what is really necessary is a way to discard the thousands of images that contain nothing but dark ocean depths. Nevertheless the first model performed as expected from the tensorboard visualizations. The model was overfit but it was still able to classify some random inputs with ~80% accuracy. It could be the lack of data provided or the nature of the data having a close similarity to each other.  

## Improving our Model:
To avoid the guessing game and improve the accuracy of the model, I want to try a different approach. This time the dataset will be constructed with "Surface features present" or "blank", unlike the previous model which attempted to differentiate the images which also included FeM. My hypothesis is that a simple model could be trained to recognize the surface features of the sea floor, then acting as the bottom layers for a new network. Another model could then use these weights, while using the top layers to determine FeM presence. 

## To be continued... 






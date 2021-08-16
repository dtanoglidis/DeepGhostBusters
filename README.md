
# DeepGhostBusters

<p align="center">
  <img  src="images/ghost.png">
</p>

Repository for the paper "DeepGhostBusters: Using Mask R-CNN to Detect and Mask Ghosting and Scattered-Light Artifacts from Optical Survey Images"\
Astronomy & Computing, submitted\
ArXiv: 

**Authors:**\
Dimitrios Tanoglidis, ...........


### Abstract 
Wide-field astronomical surveys are often plagued by the presence of reflected (often known as "ghosting artifacts" or, simply, "ghosts") and scattered-light artifacts. 
The identification and mitigation of these artifacts is crucial for rigorous astronomical analyses. 
However, the identification of ghosts and scattered-light artifacts is challenging given the complex morphology of these features and the large data volume of current and near-future surveys. 
In this work we use images from the Dark Energy Survey to train, validate, and test a Mask R-CNN model to detect and localize ghosts and scattered-light artifacts. 
We find that the ability of the Mask R-CNN model to mask affected regions is superior to that of conventional algorithms and classical CNN methods. 
We propose that a multi-step pipeline combining Mask R-CNN segmentation with a classical CNN classifier provides a powerful technique for the automated detection of ghosting and scattered-light artifacts in current and near-future surveys.


---

### Table of contents

- [Mask R-CNN](#Mask-R-CNN-implementation)
- [Datasets](#Datasets)
- [Training](#Training)
- [Inference](#Inference & Evaluation)
- [Notebook descriptions](#Notebook-descriptions)
- [Requirements](#Requirements)


---
### Mask R-CNN implementation

<p align="left">
  <img  src="images/Mask_R-CNN.png", width=450>
</p>

[Mask R-CNN](https://arxiv.org/abs/1703.06870) is a state-of-the art instance segmentation algorithm developed by Facebook's AI Research team ([FAIR](https://ai.facebook.com/)).

In this work we use the Mask R-CNN implementation developed by [Matterport](https://github.com/matterport/Mask_RCNN) on Python 3, Keras, and TensorFlow.

---
### Datasets

<p float="center">
  <img src="/images/Image_ghost.png" width="400" />
  <img src="/images/Mask_ghost.png" width="395" /> 
</p>

Training the Mask R-CNN algorithm requires a number of images and and ground truth **segmentation masks** identifying objects of interest in each image.

We use 2000,  400x400 pixel, ghost- and scattered-light-containing focal plane images, coming from the full six years of operations of the Dark Energy Survey ([DES](https://www.darkenergysurvey.org/)). 
These come from the training set used in [Chang et al., 2021](https://arxiv.org/abs/2105.10524) to train a standard CNN to distinguish clean from ghost-containing images, and are publicly available [here](https://des.ncsa.illinois.edu/releases/other/paper-data).

The masks were manually generated by eight of the authors of the present work, using the VGG Image Annotator ([VGG](https://www.robots.ox.ac.uk/~vgg/software/via/)). The annotatios are provided as `json` files containing, for each image, the coordinates of the manually created masks. During the annotation process we use classify each artifact into one of the three morphological categories `Bright`, `Faint`, `Rays`, as described in the paper. 
An example of an artifact-containing image (left) with the corresponding masks (right) is shown above (these artifacts are of the type `Rays`).

Before training, we randomly split the full dataset (images and corresponding masks) into a training set (1400 images), a validation set (300 images), and a test set (300 images), as shown in the notebook [Dataset_Split.ipynb](/Datasets/Dataset_Split.ipynb).

Inside the [Datasets](/Datasets) folder you can find as `zip` files the three sets:

- `Training_set_1.zip`, `Training_set_1.zip`,  `Training_set_3.zip`. These files, combined, constitute the training set (images only). Ww split the training set into thtree files here, because the full dataset was too large to be uploaded as a single file.
- `Validation_set.zip`, for the images in the validation set.
- `Test_set.zip`, for the images in the test set.

All the annotation (masks and classes) `json` files can be found in the folder [Annotations](/Datasets/Annotations). Specifically, we provide both the annotation files generated by each one of us (`Name_i.json`, i=1,2,3,4) as well as:

- `annotations_train.json`, for the training set.
- `annotations_val.json`, for the validation set.
- `annotations_test.json`, for the test set.

In the [Common_Core](/Datasets/Common_Core) folder, within the [Datasets](/Datasets) folder, we have a set of 50 images (`Common_core.zip`) and annotation files generated, for this common set of images, by each one of the annotators (`common_core_{name}.json`). These were used in the [Annotators_Comparison.ipynb](/Annotators_Comparison.ipynb) notebook to assess the agreement between the masks generated by the different annotator, by producing, for example, figures like the one below:

<p align="left">
  <img  src="/images/Overlap_Rays.png", width=350>
</p>

In cases where we wanted to compare with the results of the **Ray-Tracing algorithm** that is currently used by DES to identify ghost-containing CCDs in focal plane images, we used the files available [here](https://des-ops.fnal.gov:8082/exclude/) (`Ghost images` files, containing exposure numbers `expnum` and affected CCD numbers, `ccdnum`).

---
### Training

<p align="left">
  <img  src="/images/Total_Loss.png", width=450>
</p>

---
### Inference & Evaluation

---
### Notebook descriptions

Here we give brief descriptions of the contents of notebook in this repository

---
### Requirements

We ran all of our experiments in Google Colab Pro, using GPUs and High-RAM mode.


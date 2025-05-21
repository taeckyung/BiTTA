# BiTTA: Test-Time Adaptation with Binary Feedback

This is the PyTorch Implementation of "Test-Time Adaptation with Binary Feedback (ICML '25)" by 
[Taeckyung Lee](https://taeckyung.github.io/),
[Sorn Chottananurak](https://s6007541.github.io/),
[Junsu Kim](https://junsu-kim97.github.io/),
[Jinwoo Shin](https://alinlab.kaist.ac.kr/shin.html/),
[Taesik Gong*](https://taesikgong.com/), and
[Sung-Ju Lee*](https://sites.google.com/site/wewantsj/) (* Corresponding authors).

[[ OpenReview ]](https://openreview.net/forum?id=A5l37HL8vX) 
[[ arXiv ]]() 
[[ Website ]](https://nmsl.kaist.ac.kr/projects/bitta/)


## Installation Guide

1. Download or clone our repository
2. Set up a python environment using conda (see below)
3. Prepare datasets (see below)
4. Run the code (see below)

## Python Environment

We use [Conda environment](https://docs.conda.io/).
You can get conda by installing [Anaconda](https://www.anaconda.com/) first.

We share our python environment that contains all required python packages. Please refer to the `./bitta.yml` file

You can import our environment using conda:

    conda env create -f bitta.yml -n bitta

Reference: https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#creating-an-environment-from-an-environment-yml-file

## Prepare Datasets

To run our codes, you first need to download at least one of the datasets. Run the following commands:

    $ cd .                               #project root
    $ . download_cifar10c.sh             #download CIFAR10/CIFAR10-C datasets
    $ . download_cifar100c.sh            #download CIFAR100/CIFAR100-C datasets
    $ . download_pacs.sh                 #download PACS datasets
    $ . download_tiny_imagenet.sh        #download Tiny-ImageNet-C datasets

## Run

### Prepare Source model
You first need to create the directory for pre-trained weights :

    $ cd .                               #project root
    $ mkdir pretrained_weights           #create blank directory for pre-trained weights

We prepare the pre-trained and fine-tuned model for adaptation at [Google Drive link](https://drive.google.com/drive/folders/1gJt0uRVQRVML-kk6aLgLFLMFxnUJ-k4y?usp=sharing). 
Make sure that the pre-trained and fine-tuned model files are in the `./pretrained_weights` folder:
```
BiTTA
│   README.md
│   tta.sh
│   main.py
│   download_pacs.sh
|   ...
|
└───pretrained_weights
│   └───pacs
│   |   │   bitta_cp
│   |
│   └───tiny-imagenet
│       │   bitta_cp
│   
...
```
### Run Test-Time Adaptation (TTA) & Estimate Accuracy

Given source models are available, you can run TTA via:

    $. tta.sh                       #Run continual CIFAR10-C with BiTTA as default.

You can specify which dataset and which method in the script file.

Note that you must add `--active_binary` flag to enable TTA with binary feedback for TTA/active-TTA methods.

## Log

### Raw logs

In addition to console outputs, the result will be saved as a log file with the following structure: `./log/{DATASET}/{METHOD}_outdist/{TGT}/{LOG_PREFIX}_{SEED}_{DIST}/online_eval.json`

### Obtaining results

In order to print the accuracy estimation mean-absolute-errors(%) on test set, run the following commands:

    #### print the result in continual TTA setting. ####
    $ python print_acc.py --dataset pacs --target BiTTA --seed 0 1 2 --cont
   
    #### print the result in online TTA setting. ####
    $ python print_acc.py --dataset pacs --target BiTTA --seed 0 1 2  


## Tested Environment

We tested our codes under this environment.

- OS: Ubuntu 20.04.4 LTS
- GPU: NVIDIA GeForce TITAN 3090
- GPU Driver Version: 535.216.01
- CUDA Version: 12.2

# ResNet20 recreation with PyTorch

This repo is designed to be used on a google collab instance.

You can play with the hyperparameters in the `train.py` file. 
According to the original ResNet paper[[1]](#1), he author discusses the case of CIFAR10 in section 4.2 CIFAR-10 and Analysis, providing the following table:

| **output map size** | **32 x 32** | **16 x 16** | **8 x 8** |
|---------------------|-------------|-------------|-----------|
| **# layers**        | 1 + 2n      | 2n          | 2n        |
| **# filters**       | 16          | 32          | 64        |

**Table: ResNet parameters for CIFAR-10**

Choosed $n=1$ for simplicity, leading to a ResNet20 architecture.

## Data Augmentation
The model’s ability to generalize must therefore be high.
To achieve this, I have then modified the code in order to apply data augmentation. To to so, I applied random transformations to the dataset, such as shifts, zooms and horizontal flips. Similarly, to speed up the training process, the dataset is normalized. Thoses transformations are only used on the training set. The validation and testing one are not altered by it.

## Learning rate scheduler
The OneCycleLR, introduced in this paper[[2]](#2) is applied.

## Results
To arrive to the current checkpoint, the Adam Optimizer and the CrossEntropyLoss are both used. With the following hyperparameters : 
- epochs = 8
- lr = 0.01
- batch size = 5,
  a train accuracy of ~ 91% is achieved on cifar10 with a validation accuracy of 83%.As for testing accuracy, we’re at around 84%, which is a pretty good performance for a model that’s not that deep.
  
## Bibliography
<a id="1">[1]</a> : Kaiming He et al. *Deep Residual Learning for Image Recognition.* 2015. arXiv: [1512. 03385 [cs.CV]](https://arxiv.org/abs/1512.03385).

<a id="2">[2]</a> : Leslie N. Smith and Nicholay Topin. *Super-Convergence: Very Fast Training of Neu- ral Networks Using Large Learning Rates.* 2018. arXiv: [1708.07120 [cs.LG]](https://arxiv.org/abs/1708.07120).

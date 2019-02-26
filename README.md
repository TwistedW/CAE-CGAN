CAE-CGAN
======================================================

This repository contains code accompanying the paper [Image Generation via Latent Space Learning Using Improved Combination](https://doi.org/10.1016/j.neucom.2019.02.031).

The code is tested on Linux operating system with Python 2.7 or 3.x, TensorFlow 1.4.0+.

Run the model using this command:
-------------------------------------

if you want to train unsupervised version:

	python main.py --gan_type AE_GAN --dataset mnist

supervised version:
	
	python main.py --gan_type CAE_CGAN --dataset mnist

The dataset you can choose mnist or fashion-mnist.You should put the dataset in the data folder.

```
├── data
│   ├── mnist # mnist data (not included in this repo)
│   |   ├── t10k-images-idx3-ubyte.gz
│   |   ├── t10k-labels-idx1-ubyte.gz
│   |   ├── train-images-idx3-ubyte.gz
│   |   └── train-labels-idx1-ubyte.gz
│   └── fashion-mnist # fashion-mnist data (not included in this repo)
│       ├── t10k-images-idx3-ubyte.gz
│       ├── t10k-labels-idx1-ubyte.gz
│       ├── train-images-idx3-ubyte.gz
│       └── train-labels-idx1-ubyte.gz
└── 
```

If you want to try more dataset you can contact me or modify by detail structure.

#### Our model trains the results on the MNIST dataset.
*Name* | *Epoch 1* | *Epoch 10* | *Epoch 25* | GIF |
:---: | :---: | :---: | :---: | :---: |
AE-GAN | <img src = 'assets/AE-GAN/mnist/1.png' height = '200px'> | <img src = 'assets/AE-GAN/mnist/10.png' height = '200px'> | <img src = 'assets/AE-GAN/mnist/25.png' height = '200px'> | <img src = 'assets/AE-GAN/mnist/mnist.gif' height = '200px'>
CAE-CGAN | <img src = 'assets/CAE-CGAN/mnist/1.png' height = '200px'> | <img src = 'assets/CAE-CGAN/mnist/10.png' height = '200px'> | <img src = 'assets/CAE-CGAN/mnist/25.png' height = '200px'> | <img src = 'assets/CAE-CGAN/mnist/mnist.gif' height = '200px'>

#### Our model trains the results on the fashion-MNIST dataset.
*Name* | *Epoch 1* | *Epoch 10* | *Epoch 25* | GIF |
:---: | :---: | :---: | :---: | :---: |
AE-GAN | <img src = 'assets/AE-GAN/fashion-mnist/1.png' height = '200px'> | <img src = 'assets/AE-GAN/fashion-mnist/10.png' height = '200px'> | <img src = 'assets/AE-GAN/fashion-mnist/25.png' height = '200px'> | <img src = 'assets/AE-GAN/fashion-mnist/fashion-mnist.gif' height = '200px'>
CAE-CGAN | <img src = 'assets/CAE-CGAN/fashion-mnist/1.png' height = '200px'> | <img src = 'assets/CAE-CGAN/fashion-mnist/10.png' height = '200px'> | <img src = 'assets/CAE-CGAN/fashion-mnist/25.png' height = '200px'> | <img src = 'assets/CAE-CGAN/fashion-mnist/fashion-mnist.gif' height = '200px'>

#### Our model trains the results on the CIFAR-10 dataset. For CAE-CGAN we choose ship class to show.
*Name* | *Epoch 1* | *Epoch 20* | *Epoch 50* |  *Epoch 100* |
:---: | :---: | :---: | :---: | :---: |
AE-GAN | <img src = 'assets/AE-GAN/cifar10/1.png' height = '200px'> | <img src = 'assets/AE-GAN/cifar10/20.png' height = '200px'> | <img src = 'assets/AE-GAN/cifar10/50.png' height = '200px'> | <img src = 'assets/AE-GAN/cifar10/100.png' height = '200px'>
CAE-CGAN | <img src = 'assets/CAE-CGAN/cifar10/1.png' height = '200px'> | <img src = 'assets/CAE-CGAN/cifar10/20.png' height = '200px'> | <img src = 'assets/CAE-CGAN/cifar10/50.png' height = '200px'> | <img src = 'assets/CAE-CGAN/cifar10/100.png' height = '200px'>

#### Our model trains the results on the celebA dataset.
*Name* | *Black_Hair* | *Blond_Hair* 
:---: | :---: | :---: |
CAE-CGAN | <img src = 'assets/CAE-CGAN/celebA/14.png' height = '300px'> | <img src = 'assets/CAE-CGAN/celebA/15.png' height = '300px'>
*Name* | *Brown_Hair* | *Gray_Hair* 
CAE-CGAN | <img src = 'assets/CAE-CGAN/celebA/16.png' height = '300px'> | <img src = 'assets/CAE-CGAN/celebA/17.png' height = '300px'>

####  Prediction of the faces of the same person with different ages, each row corresponding to the same person.
<img src = 'assets/CAE-CGAN/celebA/23.png'>

####  Facial attributes generation.
<img src = 'assets/CAE-CGAN/celebA/22.png'>

#### Other dataset network structure.

<p align="center">
    <img src="/assets/structure1.png">
</p>

<p align="center">
    <img src="/assets/structure2.png">
</p>

email: twistedwg@hotmail.com




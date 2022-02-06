# Comp411/511 HW2

This assignment is adapted from [Stanford Course CS231n](http://cs231n.stanford.edu/).

In this assignment you will practice writing backpropagation code, and training Neural Networks and Convolutional Neural Networks. The goals of this assignment are as follows:
- understand Neural Networks and how they are arranged in layered architectures
- understand and be able to implement (vectorized) backpropagation
- implement various update rules used to optimize Neural Networks
- implement Dropout to regularize networks
- understand Batch Normalization and Layer Normalization and use for training deep networks
- understand the architecture of Convolutional Neural Networks and get practice with training these models on data
- gain experience with PyTorch


## Setup Instructions
**NOTE**: If you have already set up your environment for the first assignment, you can skip this part.
You can use the same environment in this assignment as well. However, you should still check out the Package Dependencies section.

**Installing Anaconda:** If you decide to work locally, we recommend using the free [Anaconda Python distribution](https://www.anaconda.com/download/), which provides an easy way for you to handle package dependencies. Please be sure to download the Python 3 version, which currently installs Python 3.7. We are no longer supporting Python 2.

**Anaconda Virtual environment:** Once you have Anaconda installed, it makes sense to create a virtual environment for the course. If you choose not to use a virtual environment, it is up to you to make sure that all dependencies for the code are installed globally on your machine. To set up a virtual environment, run (in a terminal)

`conda create -n comp411 python=3.7 anaconda`

to create an environment called comp411.

Then, to activate and enter the environment, run

`conda activate comp411`

To exit, you can simply close the window, or run

`conda deactivate comp411`

Note that every time you want to work on the assignment, you should run `conda activate comp411` (change to the name of your virtual env).

You may refer to [this page](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html) for more detailed instructions on managing virtual environments with Anaconda.

#### Package Dependencies
If you have `pip` installed on your system (normally conda does it by default), you may use `pip` to install the
necessary python packages conveniently. From the project root, type the following:

`pip install -r requirements.txt`

This command will install the correct versions of the dependency packages listed in the `requirements.txt` file.


## Download data:

Once you have the starter code (regardless of which method you choose above), you will need to download the CIFAR-10
 dataset. Make sure `wget` is installed on your machine before running the commands below. Run the following from the 
 `comp411_assignment2` directory:

```
cd comp411/datasets
./get_datasets.sh
```

## Start IPython:

After you have the CIFAR-10 data, you should start the IPython notebook server from the assignment2 directory, with the `jupyter notebook` command.

If you are unfamiliar with IPython, you can also refer the [IPython tutorial](https://cs231n.github.io/python-numpy-tutorial/).

## Grading
### Q1: Fully-connected Neural Network (38 points)
The IPython notebook `FullyConnectedNets.ipynb` will introduce you to our modular
 layer design, and then use those layers to implement fully-connected networks of arbitrary depth. To optimize these 
 models you will implement several popular update rules.

### Q2: Dropout (13 points)
The IPython notebook `Dropout.ipynb` will help you implement Dropout and explore
its effects on model generalization.

### Q3: Convolutional Networks (27 points)
In the IPython Notebook `ConvolutionalNetworks.ipynb` you will implement several 
new layers that are commonly used in convolutional networks.

### Q4: PyTorch on CIFAR-10 (22 points)
For this last part, you will be working in PyTorch, a popular and powerful deep learning framework.
Open up `PyTorch.ipynb`. There, you will learn how the framework works, culminating
in training a  convolutional network of your own design on CIFAR-10 to get the best performance you can.
    
## Submission

Zip (do not use RAR) the assignment folder using the format `username_studentid_assignment2.zip`.
Upload the zip file to [Blackboard](https://blackboard.ku.edu.tr/). Do not include large files in the submission (for instance data files under `./comp411/datasets/cifar-10-batches-py`).

## Notes

NOTE 1: Make sure that your homework runs successfully. Otherwise, you may get a zero grade from the assignment.

NOTE 2: There are # *****START OF YOUR CODE/# *****END OF YOUR CODE tags denoting the start and end of code sections you should fill out. Take care to not delete or modify these tags, or your assignment may not be properly graded.

NOTE 3: The assignment2 code has been tested to be compatible with python version 3.7 (it may work with other versions of 3.x, but we won’t be officially supporting them). You will need to make sure that during your virtual environment setup that the correct version of python is used. You can confirm your python version by running `python --version`.


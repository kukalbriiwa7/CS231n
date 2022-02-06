# Comp411/511 HW1

This assignment is adapted from [Stanford Course CS231n](http://cs231n.stanford.edu/).

In this assignment you will practice putting together a simple image classification pipeline, based on the k-Nearest Neighbor or the SVM/Softmax classifier. The goals of this assignment are as follows:

- understand the basic Image Classification pipeline and the data-driven approach (train/predict stages)
- understand the train/val/test splits and the use of validation data for hyperparameter tuning.
- develop proficiency in writing efficient vectorized code with numpy
- implement and apply a k-Nearest Neighbor (kNN) classifier
- implement and apply a Multiclass Support Vector Machine (SVM) classifier
- implement and apply a Softmax classifier
- implement and apply a Two layer neural network classifier
- understand the differences and tradeoffs between these classifiers
- get a basic understanding of performance improvements from using higher-level representations than raw pixels (e.g. color histograms, Histogram of Gradient (HOG) features)

## Setup Instructions

**Installing Anaconda:** If you decide to work locally, we recommend using the free [Anaconda Python distribution](https://www.anaconda.com/download/), which provides an easy way for you to handle package dependencies. Please be sure to download the Python 3 version, which currently installs Python 3.7. We are no longer supporting Python 2.

**Anaconda Virtual environment:** Once you have Anaconda installed, it makes sense to create a virtual environment for the course. If you choose not to use a virtual environment, it is up to you to make sure that all dependencies for the code are installed globally on your machine. To set up a virtual environment, run (in a terminal)

`conda create -n comp411 python=3.7 anaconda`

to create an environment called comp451.

Then, to activate and enter the environment, run

`conda activate comp411`

To exit, you can simply close the window, or run

`conda deactivate comp411`

Note that every time you want to work on the assignment, you should run `conda activate comp411` (change to the name of your virtual env).

You may refer to [this page](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html) for more detailed instructions on managing virtual environments with Anaconda.

In order to use the correct version of `scipy` for this assignment,
run the following after your first activation of the environment:

`conda install scipy=1.1.0`

## Download data:

Once you have the starter code (regardless of which method you choose above), you will need to download the CIFAR-10 dataset. Make sure `wget` is installed on your machine before running the commands below. Run the following from the assignment1 directory:

```
cd comp411/datasets
./get_datasets.sh
```

## Start IPython:

After you have the CIFAR-10 data, you should start the IPython notebook server from the assignment1 directory, with the `jupyter notebook` command.

If you are unfamiliar with IPython, you can also refer to our [IPython tutorial](https://cs231n.github.io/python-numpy-tutorial/).

## Grading

Q1: k-Nearest Neighbor classifier (30 points)

Q2: Implement a Softmax classifier (25 points)

Q3: Training a Support Vector Machine (0 points)
This is implemented for you. Please check the answer and run the code.

Q4: Three-Layer Neural Network (30 points)

Q5: Higher Level Representations: Image Features (15 points)

## Submission

Zip (do not use RAR) the assignment folder using the format `username_studentid_assignment1.zip`.
Upload the zip file to [Blackboard](https://blackboard.ku.edu.tr/). Do not include large files in the submission (for instance data files under `./comp411/datasets/cifar-10-batches-py`).

## Notes

NOTE 1: Make sure that your homework runs successfully. Otherwise, you may get a zero grade from the assignment.

NOTE 2: There are # *****START OF YOUR CODE/# *****END OF YOUR CODE tags denoting the start and end of code sections you should fill out. Take care to not delete or modify these tags, or your assignment may not be properly graded.

NOTE 3: The assignment1 code has been tested to be compatible with python version 3.7 (it may work with other versions of 3.x, but we won’t be officially supporting them). You will need to make sure that during your virtual environment setup that the correct version of python is used. You can confirm your python version by running `python --version`.

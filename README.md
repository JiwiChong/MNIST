# MNIST

This assignment focuses on the task of image classification using MNIST dataset, which
is comprised of images that are labeled with digits from 0 to 9. A deep learning algorithm
will be used to undertake the classification task. Such algorithm will be deployed later on
via an API for other users to interact with and test its functionality by inputting an image.
The algorithm of choice is LeNet-5, a simple yet an effective variant of Convolutional Neural
Network, which automatically labels a given MNIST image with a digit label. 

LeNet-5 was built using PyTorch. Throughout the training phase, the training and validation loss
values are logged in on MLFlow. At MLFlow, users can verify the progress of such loss values 
to see whether they drop as the model gets trained and its parameters are tuned to fit the images
to their corresponding labels. 

Prior to execute the training phase of LeNet-5, the user must run the following command in command
prompt in order to create an URI via which he can observe the logged in loss values.


##Install the required libraries:
```python
pip install requirements.txt
```

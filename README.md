# API Service with MNIST

This assignment focuses on multiple tasks involving the MNIST dataset, which
is comprised of images that are labeled with digits from 0 to 9. The tasks include, implementation and training of a
deep learning model for the classification task, and testing of the model
to observe its prediction on newly seen data. Throughout these processes, the model will be registered to MLFlow 
and its information will be stored within certain folders in the repository. The model will ultimately be deployed later on
via an API for other users to interact with and test its functionality.

### Framework of the Assignment

<div align="center">
<img src="https://github.com/user-attachments/assets/928aca97-6b4a-4d98-aabd-d2d4f3967a65" width=100% height=100%>
</div><br />

The algorithm of choice is LeNet-5, a simple yet an effective variant of Convolutional Neural
Network, which automatically labels a given MNIST image with a digit label. 

<div align="center">
<img src="https://github.com/user-attachments/assets/bb1817ef-743d-4111-b2a8-8e63bf15cbca" width=90% height=90%>
</div><br />

LeNet-5 was built using PyTorch. Throughout the training phase, the training and validation loss
values are logged in on MLFlow. At MLFlow, users can verify the progress of such loss values 
to see whether they drop as the model gets trained and its parameters are tuned to fit the images
to their corresponding labels. 

Prior to executing the training phase of LeNet-5, the user must run the following command in command
prompt in order to create an URI via which he can access MLFlow experiments to observe the logged in loss values 
and information of the model:
```python
mlflow server --host 127.0.0.1 --port 8080
```
### conda environment creation
```python
conda create -n mediwhale_mnist python=3.8
```
### Install the required libraries:
```python
conda install pytorch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 pytorch-cuda=11.8 -c pytorch -c nvidia
pip install requirements.txt
```
### Train:
```python
python main_mlflow.py --batch_size 32 --learning_rate 0.001 --model_name lenet5 --num_epochs 10 --data_name mnist --n_classes 10 --run_num (# of run (if first, input is 1))
```
After the training phase is completed, LeNet-5's summary is saved in __"/model_summary/lenet5/"__ and its format is converted to ONNX. 
ONNX is a valuable tool that provides a common file format for users to change AI models from one framework to another, allowing
users to configure and run them with other tools and compilers. Throughout the training phase, the summary of the model and the
model, with weights that delivered the lowest epoch validation loss, are saved. Once model is finished training, its ONNX version
is registered to MLFlow to be utilized for deployment and the information about the tracked experiment entry is returned.
**NOTE: Make sure the directories are set according to the local machine!**

### Test:
```python
python eval.py --batch_size 32 --model_name lenet5 --data_name mnist --n_classes 10 --run_num (# of run (if first, input is 1))
```
When tested, LeNet-5 achieved both **Accuracy** and **F1-Scores** of **99%**. 
It achieved such near perfect scores when it was trained with Cross-Entropy Loss function and the 
following hyperparameters:<br />

**epochs = 10<br />
  learning rate = 0.001<br />
  batch size = 32<br />
  Optimizer = Adam<br />**<br />
  
In addition, information about the registered model is saved in __"/registered_models/lenet5/run_(given run number)/"__

The API that was used for deployment in this assignment is Flask. It provides an interaction service in which a 
user inputs any test MNIST image and its predicted label is returned. 

This deployment service is built using Docker as well so that any user can carry the MNIST classifying software
into a container and use it on his machine. Prior to running the commands below, make sure they are ran within
the directory in which the Dockerfile is located in. __pytorch/pytorch:2.3.1-cuda11.8-cudnn8-runtime__ tag is pulled
from Docker Hub. This is the tag which ensures that the container will have PyTorch 2.3.1 and Nvidia Cuda of version 11.8
already available. 

### Docker
#### Run Dockerfile:
```python
docker build -f Dockerfile -t pytorch/pytorch:2.3.1-cuda11.8-cudnn8-runtime .
```
#### Build docker image:
```python
docker build -t (image name) .
	Ex: docker build -t mediwhale_task .
```
#### Build docker container with the image created above:
```python
docker run --name (container name) --gpus all -it (image name)
	Ex: docker run --name mediwhale --gpus all -it mediwhale_task
```

Alternatively, the deployment can be executed using docker-compose by running docker-compose.yml file
that runs every command on Dockerfile. It also creates the image and container in which every relevant
library, environment configuration, and deployment is executed.

```python
docker compose up -d --build
```
Finally, the user can interact with the MNIST predictor via: http:/localhost:(port number give at EXPOSE part of Dockerfile).
In this case, if the user happens to have ran Dockerfile first, and he needs to create a different container for the same task,
he must make sure that a new port number is used. At the "port" part of docker-compose.yml, the user can insert a new 
port number (Ex: 9090:8080). In this case, the user can interact with the MNIST predictor via: http:/localhost:9090. 

Lastly, the user can directly run the following command in his local machine to utilize the API service for MNIST
classification:

```python
python app.py
```
Once this finishes running, access the given URI and the following screen should pop out:

<div align="center">
<img src="https://github.com/user-attachments/assets/b50817d8-cc78-4cdd-a76a-2eff89595cc6" width=100% height=100%>
</div><br />

Make sure that the "source" and "store_location" addresses are set to the one that belongs to the local
machine, not the one for Docker. This is the directory in which the registered Onnx model is saved in. 

<div align="center">
<img src="https://github.com/user-attachments/assets/e23168a5-e4fc-4e02-a3d6-e966118fcb16" width=70% height=70%>
</div><br />

That's it! Feel free to test as many new test MNIST images as you would like! All you have to do is select any image file
from your machine and click on the blue "Predict Image" button. Test image files are in __"/dataset/MNIST/test_images/"__.

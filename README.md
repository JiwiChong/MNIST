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
prompt in order to create an URI via which he can observe the logged in loss values:
```python
mlflow server --host 127.0.0.1 --port 8080
```
### github repository cloning and conda environment creation
```python
git clone Assignment.git
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
### Test:
```python
python eval.py --batch_size 32 --model_name lenet5 --data_name mnist --n_classes 10 --run_num (# of run (if first, input is 1))
```
When tested, LeNet-5 achieved both **Accuracy** and **F1-Scores** of **99%**. 

After the training phase is completed, LeNet-5's format is converted to ONNX. ONNX is a valuable tool
that provides a common file format for users to change AI models from one framework to another, allowing
users to configure and run them with other tools and compilers. Afterwards, the ONNX model is registered
to MLFlow to be utilized for deployment.

The API that was used for deployment in this assignment is Flask. It provides an interaction service in which a 
user inputs any of the test MNIST image and returns the predicted label of it. 

This deployment service is built using Docker as well so that any user can carry the MNIST classifying software
into a container and use it on his machine. Prior to running the commands below, make sure they are ran within
the directory in which the Dockerfile is located in.
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
that runs every command on Dockerfile given the docker image and container names for which it was created.

```python
docker compose up -d --build
```

import os
import cv2
import numpy as np
import torchvision.transforms as transforms
import mlflow
import onnxruntime 
from flask import Flask, render_template, request

model_uri = "models:/mnist_lenet5_model/1"  # Version 1 of the model
registered_model = mlflow.onnx.load_model(model_uri)
registered_model.SerializeToString()
onnx_session = onnxruntime.InferenceSession(registered_model.SerializeToString())

app = Flask(__name__)

@app.route('/', methods=['GET'])
def render_mnist():
    return render_template('display.html')  

@app.route('/', methods=['POST'])
def predict_and_identify():
    # image_file = input(str('Insert image file name you would like to use'))
    image_file = request.files['imagefile']
    image_tensor = cv2.imread(os.path.join('./dataset/MNIST/test_images/', image_file.filename))

    cv2.imwrite(f'static/images/{image_file.filename}', image_tensor)
    # image_tensor = cv2.imread(os.path.join('./dataset/MNIST/test_images/00003.jpg'))
    image_tensor_gray = cv2.cvtColor(image_tensor, cv2.COLOR_BGR2GRAY)
    input_img_transform = transforms.Compose([
                        transforms.ToPILImage(),
                        transforms.Pad(2),  # Add 2 pixels of padding to each side of the 28x28 image to make it 32x32
                        transforms.ToTensor()
                        ])

    input_image_ = input_img_transform(image_tensor_gray).unsqueeze(0)
    inputs = {"input": input_image_.numpy()}
    
    onnx_outputs = onnx_session.run(None, inputs)
    classification = f'The predicted digit is: {np.argmax(onnx_outputs[0])}'

    return render_template('display.html', prediction=classification, user_image = image_file.filename)

if __name__ == '__main__':
     app.run(host='0.0.0.0', port=8080)

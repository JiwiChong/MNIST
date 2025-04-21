import os
import onnx
import mlflow
import pickle
from mlflow.models import infer_signature
import torch
import argparse
import numpy as np
from models import LeNet5
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision import datasets
from sklearn.metrics import accuracy_score, f1_score

def predict_with_metric_scores(model, test_dataloaders):
    model.eval()
    all_y, all_y_hat = [], []
    for batch in test_dataloaders:
        inputs, labels = batch[0].to(device), batch[1].to(device)
        with torch.no_grad():
            # Add batch to GPU
            outputs = model(inputs)
            y = labels.detach().cpu().numpy().tolist()
            y_pred = outputs.detach().cpu().numpy().tolist()
            all_y.extend(y)
            all_y_hat.extend(y_pred)

    final_all_y_pred = np.argmax(all_y_hat, axis=1)
    test_f1 = round(f1_score(all_y, final_all_y_pred, average='weighted'), 2)
    print(f'Test Accuracy is: {round(accuracy_score(all_y, final_all_y_pred), 2)}')
    print(f'Test F1 Score is: {test_f1}')

    return {'test_acc': round(accuracy_score(all_y, final_all_y_pred), 2),'test_f1':test_f1}


def onnx_registry(args, sample_x, metric_dict):
    onnx_model = onnx.load(os.path.join(f'./saved_models/{args.model_name}/best_onnx_num_{args.run_num}.onnx'))
    with mlflow.start_run() as run:
        signature = infer_signature(sample_x.numpy(), model(sample_x.to(device)).cpu().detach().numpy())

        model_info = mlflow.onnx.log_model(onnx_model, 'model', signature=signature) # watch out for the middle one!!!
        # Optionally, log parameters or metrics
        mlflow.log_metric("test_accuracy", metric_dict['test_acc'])
        mlflow.log_metric("test_accuracy", metric_dict['test_f1'])
        print('Info of the registered model:')
        print(mlflow.get_run(model_info.run_id).info)
        pickle.dump(mlflow.get_run(model_info.run_id).info, open(f'./registered_models/{args.model_name}/run_{args.run_num}/model_info.pickle', 'wb'))

    model_uri = f"runs:/{run.info.run_id}/model"
    mlflow.register_model(model_uri, f"mnist_{args.model_name}_model_run_{args.run_num}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluation of AI model on MNIST')
    parser.add_argument('--batch_size', type=int, default=8, help='batch size')
    parser.add_argument('--model_name', type=str, help='algorithm')
    parser.add_argument('--data_name', type=str, help='Dataset to be used')
    parser.add_argument('--n_classes', type=int, help='number of labels for classification')
    parser.add_argument('--run_num', type=int, help='run number')
    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if args.data_name == 'mnist':
        transform = transforms.Compose([
                    transforms.Pad(2),  # Add 2 pixels of padding to each side of the 28x28 image to make it 32x32
                    transforms.ToTensor()
                    ])
        test_dataset = datasets.MNIST(root="./dataset/", train=False, download=False, transform=transform)
        test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)


    if args.model_name == 'lenet5':
        model = LeNet5(args).to(device)

    sample_batch_x, sample_batch_labels = next(iter(test_dataloader))
    sample_x = sample_batch_x[0,:,:,:].unsqueeze(0)

    model.load_state_dict(torch.load(f'./saved_models/{args.model_name}/best_model_num_{args.run_num}.pt', map_location=device)['model_state_dict'])
    metric_dictionary = predict_with_metric_scores(model, test_dataloader)
    onnx_registry(args, sample_x, metric_dictionary)


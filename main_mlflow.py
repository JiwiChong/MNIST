import os 
import gc
import argparse
import torch
import torch.nn as nn
from torch import optim
from tqdm import tqdm
import numpy as np
import torchvision.transforms as transforms
from torchvision import datasets
from models import LeNet5
import mlflow
from torchinfo import summary
from torchmetrics import Accuracy
from utils import save_checkpoint, save_plots

def run(args, model, train_dataloader, valid_dataloader, test_dataloader):
    loss_func = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    metric_fn = Accuracy(task="multiclass", num_classes=10).to(device)

    client = mlflow.MlflowClient()

    mlflow.set_tracking_uri("http://127.0.0.1:8080")
    mlflow.set_experiment("/my-mnist-experiment")
        
    with mlflow.start_run():
        params = {
            "epochs": args.num_epochs,
            "learning_rate": args.learning_rate,
            "batch_size": args.batch_size,
            "Number of Run": args.run_num,
            "loss_function": loss_func.__class__.__name__,
            "metric_function": metric_fn.__class__.__name__,
            "optimizer": "Adam",
        }
        # Log training parameters.
        mlflow.log_params(params)

        # Log model summary.
        with open(f"./model_summary/{args.model_name}/model_summary_run_{args.run_num}.txt", "w", encoding="utf-8") as f:
            f.write(str(summary(model)))
        mlflow.log_artifact(f"./model_summary/{args.model_name}/model_summary_run_{args.run_num}.txt")

        best_loss = np.inf
        best_path = f'./saved_models/{args.model_name}/run_{args.run_num}/best_model_num_{args.run_num}.pt'
 
        print('Training started!')

        epoch_train_loss = []
        epoch_valid_loss = []
        # Training loop
        for epoch in tqdm(range(args.num_epochs)):
            model.train()

            b_train_loss = []
            batch = 0
            for i, (inputs, labels) in enumerate(train_dataloader):
                inputs, labels = inputs.to(device), labels.to(device)
                
                optimizer.zero_grad()    
                model_outputs = model(inputs)

                new_label_tensor = np.zeros((len(labels), args.n_classes))
                for i, e in enumerate(list(labels)):
                    new_label_tensor[i,int(e)] = 1
                labels__ = torch.tensor(new_label_tensor, dtype=torch.float32)

                loss = loss_func(model_outputs, labels__.to(device))
                b_train_loss.append(loss.item())
                batch += 1

                del inputs
                del labels__

                torch.cuda.empty_cache()
                gc.collect()
                
                loss.backward()
                optimizer.step()
                
            epoch_train_loss.append(np.mean(b_train_loss))
            mlflow.log_metric("training_loss", f"{np.mean(b_train_loss):3f}", step=(batch // 100))
            
            print('Epoch: {}'.format(epoch+1))
            print('Training Loss: {}'.format(np.mean(b_train_loss)))

            model.eval()

            with torch.no_grad():
                b_valid_loss = []
                batch = 0 
                for i, (inputs, labels) in enumerate(valid_dataloader):
                    inputs = inputs.to(device)

                    model_outputs = model(inputs)
                    new_label_tensor = np.zeros((len(labels), args.n_classes))
                    for i, e in enumerate(list(labels)):
                        new_label_tensor[i,int(e)] = 1
                    labels__ = torch.tensor(new_label_tensor, dtype=torch.float32)

                    val_loss = loss_func(model_outputs, labels__.to(device))
                    b_valid_loss.append(val_loss.item())
                    batch += 1

                print('Validation Loss {}'.format(np.mean(b_valid_loss)))
                print('-' * 40)
                epoch_valid_loss.append(np.mean(b_valid_loss))
                mlflow.log_metric("validation_loss", f"{np.mean(b_valid_loss):3f}", step=(batch // 100))
                
                if np.mean(b_valid_loss) < best_loss:
                    best_loss = np.mean(b_valid_loss)
                    save_checkpoint(best_path, model, np.mean(b_valid_loss))

            

        sample_img, _ = next(iter(test_dataloader))
        model.load_state_dict(torch.load(best_path, map_location=device)['model_state_dict'])
        torch.onnx.export(model,
                    sample_img.to(device),
                    f'./saved_models/{args.model_name}/best_onnx_num_{args.run_num}.onnx',
                    export_params=True,
                    verbose=False,              
                    input_names=['input'],     
                    output_names=['output'])
        
        print("Training complete!")
        data = client.get_run(mlflow.active_run().info.run_id).data
        print(data)
        save_plots(epoch_train_loss, epoch_valid_loss, args)
 
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Time Series forecasting of device data')
    parser.add_argument('--batch_size', type=int, help='batch size')
    parser.add_argument('--learning_rate', type=float, default=0.001,help='learning rate')
    parser.add_argument('--model_name', type=str, help='algorithm')
    parser.add_argument('--num_epochs', type=int, help='Num of epochs')
    parser.add_argument('--data_name', type=str, help='Dataset to be used')
    parser.add_argument('--n_classes', type=int, help='number of labels for classification')
    parser.add_argument('--run_num', type=int, help='run number')
    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    plot_path= f'./plots/{args.model_name}/run_{args.run_num}/'
    model_path= f'./saved_models/{args.model_name}/run_{args.run_num}/'
    model_info_path = f'./registered_models/{args.model_name}/run_{args.run_num}/'

    writer = None
    if not os.path.exists(path=plot_path) and not os.path.exists(path=model_path) and not os.path.exists(path=model_path):
        os.mkdir(plot_path) 
        os.mkdir(model_path) 
        os.mkdir(model_info_path)
    else:
        print(f"The directories {plot_path}, {model_path}, and {model_info_path} already exist.")

    if args.data_name == 'mnist':
        transform = transforms.Compose([
                    transforms.Pad(2),  # Add 2 pixels of padding to each side of the 28x28 image to make it 32x32
                    transforms.ToTensor()
                    ])
        train_dataset = datasets.MNIST(root="./dataset/", train=True, download=False, transform=transform)
        train_set, val_set = torch.utils.data.random_split(train_dataset, [50000, 10000])
        
        # Create the data loader
        train_dataloader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
        valid_dataloader = torch.utils.data.DataLoader(val_set, batch_size=args.batch_size, shuffle=False)
        
        test_dataset = datasets.MNIST(root="./dataset/", train=False, download=False, transform=transform)
        test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)

    if args.model_name == 'lenet5':
        model = LeNet5(args).to(device)

    run(args, model, train_dataloader, valid_dataloader, test_dataloader)


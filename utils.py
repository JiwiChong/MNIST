import torch
import matplotlib.pyplot as plt

def save_checkpoint(save_path, model, loss):
    if save_path == None:
        return

    state_dict = {'model_state_dict': model.state_dict(),
                  'val_loss': loss}

    torch.save(state_dict, save_path)
    print(f'Model saved to ==> {save_path}')


def save_plots(epoch_train_loss, epoch_valid_loss, args):
    plt.figure(figsize=(14, 5))

  # Accuracy plot
    plt.subplot(1, 2, 1)
    train_loss_plot, = plt.plot(args.num_epochs, epoch_train_loss, 'r')
    val_loss_plot, = plt.plot(args.num_epochs, epoch_valid_loss, 'b')
    plt.title('Training and Validation Loss')
    plt.legend([train_loss_plot, val_loss_plot], ['Training Loss', 'Validation Loss'])
    plt.savefig(f'./plots/{args.model_name}/run_{args.run_num}/loss_plots.jpg')
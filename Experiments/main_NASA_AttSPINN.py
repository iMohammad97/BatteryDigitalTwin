from dataloader.dataloaderForSoc import NASAdata
from Model.AttSPINN import count_parameters, AttSPINN as SPINN
import argparse, os
import numpy as np
from sklearn.metrics import mean_squared_error
import torch
import matplotlib.pyplot as plt
import numpy as np
import re
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

def calc_rmse(path):
    pred_label = np.load(os.path.join(path, "pred_label.npy"))
    true_label = np.load(os.path.join(path, "true_label.npy"))
    return np.sqrt(mean_squared_error(true_label, pred_label))

def load_data(args):
    # ---- change: load *all* batteries & modes
    # files = [
    # "charge/B0005/B0005_2.csv"    
    # ]

    data = NASAdata(root=args.data_root, args=args)
    loader_dict = data.read_one_batch(mode='charge', batch='B0005') 
    return {
        'train': loader_dict['train'],
        'valid': loader_dict['valid'],
        'test':  loader_dict['test'],
    }

    # return {
    #     'train': loader_dict['train_2'],
    #     'test' : loader_dict['valid_2'],
    # }

def load_loss_history(log_path):
    """
    Parse training and validation losses from a standard AttSPINN logging.txt file.
    Returns (epochs, train_losses, valid_losses).
    """
    train_losses = []
    valid_losses = []
    epochs = []
    with open(log_path, 'r') as f:
        for line in f:
            # match train loss lines: "[Train] epoch:1, ..., total loss:0.123456"
            m_train = re.search(r'\[Train\].*total loss:([0-9\.]+)', line)
            if m_train:
                train_losses.append(float(m_train.group(1)))
                epochs.append(len(epochs) + 1)
            # match validation MSE lines: "[Valid] epoch:1, MSE: 0.012345"
            m_valid = re.search(r'\[Valid\].*MSE:\s*([0-9\.]+)', line)
            if m_valid:
                valid_losses.append(float(m_valid.group(1)))
    return epochs, train_losses, valid_losses

def plot_loss_curves(log_path, save_folder):
    epochs, train_losses, valid_losses = load_loss_history(log_path)
    plt.figure()
    plt.plot(epochs, train_losses, label='Train Loss')
    plt.plot(epochs[:len(valid_losses)], valid_losses, label='Valid MSE')
    plt.xlabel('Epoch')
    plt.ylabel('Loss / MSE')
    plt.title('Training and Validation Curves')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_folder, 'loss_curves.png'))
    plt.show()

def plot_parity_over_time(
    save_folder,
    true_fname='true_label.npy',
    pred_fname='pred_label.npy',
    out_fname='soc_over_time.png',
    title='SoC: True vs Predicted Over Time'
):
    true = np.load(os.path.join(save_folder, true_fname)).flatten()
    pred = np.load(os.path.join(save_folder, pred_fname)).flatten()
    time = np.arange(len(true))

    plt.figure()
    plt.plot(time, true, label='True SoC')
    plt.plot(time, pred, label='Predicted SoC', alpha=0.7)
    plt.xlabel('Timestep')
    plt.ylabel('SoC (normalized)')
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_folder, out_fname))
    plt.show()


def plot_residual_hist(
    save_folder,
    true_fname='true_label.npy',
    pred_fname='pred_label.npy',
    out_fname='residual_histogram.png',
    title='Residual Distribution'
):
    true = np.load(os.path.join(save_folder, true_fname))
    pred = np.load(os.path.join(save_folder, pred_fname))
    residuals = pred - true

    plt.figure()
    plt.hist(residuals, bins=50, edgecolor='black')
    plt.xlabel('Residual (Predicted - True)')
    plt.ylabel('Count')
    plt.title(title)
    plt.tight_layout()
    plt.savefig(os.path.join(save_folder, out_fname))
    plt.show()


def main():
    args = get_args()

    for exp in range(4):
        # each experiment gets its own folder
        save_folder = os.path.join(args.save_root, f"Experiment{exp+1}")
        os.makedirs(save_folder, exist_ok=True)
        args.save_folder = save_folder
        args.log_dir     = 'logging.txt'

        print(f"\n--- Experiment {exp+1} ---")
        print("Loading all NASA data...")
        dataloader = load_data(args)

        # infer feature-dim from the first batch
        x1_sample, _, _, _ = next(iter(dataloader['train']))
        x_dim = x1_sample.shape[1]

        architecture_args = {
            "solution_u_subnet_args": {
                "output_dim": 16,
                "layers_num": 5,
                "hidden_dim": 15,
                "dropout": 0,
                "activation": "leaky-relu"
            },
            "dynamical_F_subnet_args": {
                "output_dim": 16,
                "layers_num": 5,
                "hidden_dim": 15,
                "dropout": 0,
                "activation": "leaky-relu"
            },
            "attn_embed_dim_u": 16,
            "attn_heads_u": 2,
            "attn_embed_dim_F": 16,
            "attn_heads_F": 2
        }

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        model = SPINN(args, x_dim=x_dim, architecture_args=architecture_args).to(device)
        count_parameters(model)

        print("Training on *all* NASA batteries...")
        model.Train(
            trainloader=dataloader['train'],
            validloader=dataloader['valid'],
            testloader=dataloader['test']
        )

        rmse = calc_rmse(save_folder)
        print(f"Experiment {exp+1} finished; RMSE = {rmse:.4f}")

        # Run inference on the *full* dataset (test_3) and save
        true_all, pred_all = model.Test(dataloader['test_3'])
        np.save(os.path.join(save_folder, 'true_all.npy'), true_all)
        np.save(os.path.join(save_folder, 'pred_all.npy'), pred_all)

        experiment_folder = f"results/NASA/Experiment{exp+1}"
        log_file = os.path.join(experiment_folder, 'logging.txt')

        # Plot and save figures
        plot_loss_curves(log_file, experiment_folder)

        # test‐20% plots
        plot_parity_over_time(
            experiment_folder,
            true_fname='true_label.npy',
            pred_fname='pred_label.npy',
            out_fname='soc_test_over_time.png',
            title='SoC Over Time (20% Test Split)'
        )
        
        plot_residual_hist(
            experiment_folder,
            true_fname='true_label.npy',
            pred_fname='pred_label.npy',
            out_fname='residual_test_hist.png',
            title='Residuals on 20% Test Split'
        )

        # full‐dataset plots
        plot_parity_over_time(
            experiment_folder,
            true_fname='true_all.npy',
            pred_fname='pred_all.npy',
            out_fname='soc_full_over_time.png',
            title='SoC Over Time (Full Dataset)'
        )
        plot_residual_hist(
            experiment_folder,
            true_fname='true_all.npy',
            pred_fname='pred_all.npy',
            out_fname='residual_full_hist.png',
            title='Residuals on Full Dataset'
        )
def get_args():
    parser = argparse.ArgumentParser('Hyperparameters for NASA data')
    parser.add_argument('--data_root', type=str,
                        default='data/NASA data',
                        help='root folder containing charge/ and discharge/ subfolders')
    parser.add_argument('--save_root', type=str,
                        default='results/NASA',
                        help='base directory to save experiment outputs')
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--normalization_method', type=str,
                        default='min-max', choices=['min-max','z-score'])
    # training schedule
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--warmup_epochs', type=int, default=30)
    parser.add_argument('--warmup_lr', type=float, default=0.002)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--final_lr', type=float, default=0.0002)
    parser.add_argument('--lr_F', type=float, default=0.001)
    # loss weights
    parser.add_argument('--alpha', type=float,
                        default=0.08549401305482651)
    parser.add_argument('--beta', type=float,
                        default=6.040426381151848)
    # logging
    parser.add_argument('--log_dir', type=str, default='logging.txt')
    return parser.parse_args()

if __name__ == '__main__':
    main()

import os
import argparse
import numpy as np
from sklearn.metrics import mean_squared_error
from dataloader.dataloader import TJUdata
from Model.AttSPINN import AttSPINN as SPINN
from Model.AttSPINN import count_parameters
from bayes_opt import BayesianOptimization

os.environ['CUDA_VISIBLE_DEVICES'] = '0'


def load_TJU_data(args, small_sample=None):
    """
    Build {train, valid, test} DataLoaders for TJU, using *dataloader2*.
    The logic mirrors your old PINN script:

    • If args.in_same_batch == True:
        – choose the batch index (0,1,2) in args.batch
        – cells whose (1‑based) index has units‑digit 5 or 9 → **test**
        – everything else → **train/valid** (dataloader handles split)

    • Otherwise (train/test batches differ):
        – args.train_batch and args.test_batch specify which batch folders
          to use for train and test, respectively.
    """
    root = 'data/TJU data'
    data = TJUdata(root=root, args=args)

    train_list, test_list = [], []

    if args.in_same_batch:
        batch_folders = ['Dataset_1_NCA_battery',
                         'Dataset_2_NCM_battery',
                         'Dataset_3_NCM_NCA_battery']
        batch_root = os.path.join(root, batch_folders[args.batch])
        files = sorted(os.listdir(batch_root))          # deterministic order
        # TJU rule: units digit 5 or 9 → test
        for idx, f in enumerate(files):
            cid = idx + 1      # 1‑based
            full_path = os.path.join(batch_root, f)
            if cid % 10 in (5, 9):
                test_list.append(full_path)
            else:
                train_list.append(full_path)
    else:
        batch_folders = ['Dataset_1_NCA_battery',
                         'Dataset_2_NCM_battery',
                         'Dataset_3_NCM_NCA_battery']
        train_root = os.path.join(root, batch_folders[args.train_batch])
        test_root  = os.path.join(root, batch_folders[args.test_batch])
        train_list = [os.path.join(train_root, f) for f in os.listdir(train_root)]
        test_list  = [os.path.join(test_root,  f) for f in os.listdir(test_root)]

    if small_sample is not None:
        train_list = train_list[:small_sample]

    train_loader = data.read_all(specific_path_list=train_list)
    test_loader  = data.read_all(specific_path_list=test_list)

    return {
        'train': train_loader['train_2'],
        'valid': train_loader['valid_2'],
        'test':  test_loader['test_3']
    }



def get_args():
    parser = argparse.ArgumentParser('SPINNtraining – TJU dataset')

    # dataset / split
    parser.add_argument('--data', type=str, default='TJU')
    parser.add_argument('--in_same_batch', action='store_true', default=True,
                        help='train/test split inside the same batch (default)')
    parser.add_argument('--batch', type=int, default=0, choices=[0, 1, 2],
                        help='batch index if in_same_batch')
    parser.add_argument('--train_batch', type=int, default=0, choices=[0, 1, 2])
    parser.add_argument('--test_batch',  type=int, default=1, choices=[0, 1, 2])

    # dataloader
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--normalization_method', type=str, default='min-max')

    # scheduler / optimisation
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--early_stop', type=int, default=35)
    parser.add_argument('--warmup_epochs', type=int, default=30)
    parser.add_argument('--warmup_lr',  type=float, default=0.002)
    parser.add_argument('--lr',         type=float, default=0.01)
    parser.add_argument('--final_lr',   type=float, default=0.0002)
    parser.add_argument('--lr_F',       type=float, default=0.001)

    # loss weights
    parser.add_argument('--alpha', type=float, default=1.2423804350752465)
    parser.add_argument('--beta',  type=float, default=0.10931976381057806)

    # paths
    parser.add_argument('--log_dir',    type=str, default='logging.txt')
    parser.add_argument('--save_folder', type=str,
                        default='results of reviewer/SPINN/TJU results')

    return parser.parse_args()


def build_arch_args():
    return {
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
        "spinn_enabled": {"solution_u": True, "dynamical_F": True},
        "attn_embed_dim_u": 16,
        "attn_heads_u": 2,
        "attn_embed_dim_F": 16,
        "attn_heads_F": 2
    }

def main():
    args = get_args()
    arch_args = build_arch_args()

    batches = [0, 1, 2]          # three TJU batches
    n_exp   = 10                 # experiments per batch

    # ------------------------------------------------------------------
    # keep *immutable* root to build paths from
    # ------------------------------------------------------------------
    SAVE_ROOT = args.save_folder     # e.g. 'results of reviewer/SPINN/TJU results'

    for b in batches:
        if args.in_same_batch:
            args.batch = b

        # --------------------------------------------------------------
        # create /batch-folder/ once (not per experiment)
        #   • same-batch split  :  "0-0"
        #   • cross-batch split :  "{train_batch}-{test_batch}"
        # --------------------------------------------------------------
        if args.in_same_batch:
            batch_dir = f'{SAVE_ROOT}/{b}-{b}'
        else:
            batch_dir = f'{SAVE_ROOT}/{args.train_batch}-{args.test_batch}'
        os.makedirs(batch_dir, exist_ok=True)

        for exp in range(1, n_exp + 1):
            save_folder = f'{batch_dir}/Experiment{exp}'
            os.makedirs(save_folder, exist_ok=True)

            # set paths for the current run
            args.save_folder = save_folder
            args.log_dir     = 'logging.txt'

            dataloaders = load_TJU_data(args)
            model = SPINN(args, x_dim=17, architecture_args=arch_args).cuda()
            print(f'[Batch {b}] Experiment {exp}: '
                  f'{count_parameters(model):,} params')

            model.Train(trainloader=dataloaders["train"],
                        validloader=dataloaders["valid"],
                        testloader=dataloaders["test"])


# ------------------------------------------------------------------
# helper to load the resulting RMSE from a saved experiment folder
# ------------------------------------------------------------------
def calc_rmse_folder(folder):
    """Loads pred_label.npy / true_label.npy and returns RMSE."""
    pred = np.load(os.path.join(folder, 'pred_label.npy'))
    true = np.load(os.path.join(folder, 'true_label.npy'))
    return np.sqrt(mean_squared_error(true, pred))


# ------------------------------------------------------------------
# Bayesian‐Optimization routine
# ------------------------------------------------------------------
def find_best_alpha_beta(init_points=5, n_iter=25):
    """
    Run Bayesian optimization over alpha, beta ∈ [0,50] for alpha and [0,100] for beta,
    returning the pair that *maximizes* negative RMSE
    (i.e. minimizes RMSE) averaged over your 3 batches.
    """
    args_proto = get_args()                 # argument template
    arch_args   = build_arch_args()
    SAVE_ROOT   = args_proto.save_folder
    batches     = [0,1,2]

    def objective(alpha, beta):
        # for each candidate (alpha, beta), train one experiment on each batch
        rmses = []
        for b in batches:
            # reset args per batch
            args = get_args()
            args.__dict__.update(vars(args_proto))    # carry over non-split args
            args.alpha = alpha
            args.beta  = beta

            # split
            if args.in_same_batch:
                args.batch = b
                batch_dir = f'{SAVE_ROOT}/{b}-{b}'
            else:
                batch_dir = f'{SAVE_ROOT}/{args.train_batch}-{args.test_batch}'
            os.makedirs(batch_dir, exist_ok=True)

            # only one experiment per batch
            exp_folder = os.path.join(batch_dir, 'Experiment1')
            os.makedirs(exp_folder, exist_ok=True)

            # set output paths
            args.save_folder = exp_folder
            args.log_dir     = 'logging.txt'

            # load data, build & train
            loaders = load_TJU_data(args)
            model   = SPINN(args, x_dim=17, architecture_args=arch_args).cuda()
            model.Train(trainloader=loaders['train'],
                        validloader=loaders['valid'],
                        testloader=loaders['test'])

            # record RMSE
            rmses.append(calc_rmse_folder(exp_folder))

        # We want to maximize the target, so return negative rmse
        return -np.mean(rmses)

    # configure optimizer
    optimizer = BayesianOptimization(
        f=objective,
        pbounds={'alpha': (0,50), 'beta': (0,100)},
        random_state=42,
        verbose=2
    )

    # run
    optimizer.maximize(init_points=init_points, n_iter=n_iter)

    # report
    print("-----> Best parameters found:")
    print(optimizer.max)

    return optimizer.max

# ------------------------------------------------------------------
if __name__ == '__main__':
    # to do a standard train:
    main()

    # OR to run bayesian search of best params instead of main(), uncomment:
    # find_best_alpha_beta(init_points=10, n_iter=50)

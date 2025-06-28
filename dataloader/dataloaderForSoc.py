import pandas as pd
import numpy as np
import torch
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
import os
import random
from sklearn.model_selection import train_test_split
from utils.util import write_to_txt

class DF():
    def __init__(self,args):
        self.normalization = True
        self.normalization_method = args.normalization_method # min-max, z-score
        self.args = args

    def _3_sigma(self, Ser1):
        '''
        :param Ser1:
        :return: index of outliers
        '''
        rule = (Ser1.mean() - 3 * Ser1.std() > Ser1) | (Ser1.mean() + 3 * Ser1.std() < Ser1)
        index = np.arange(Ser1.shape[0])[rule]
        return index

    def delete_3_sigma(self,df):
        '''
        :param df: DataFrame
        :return: cleaned DataFrame
        '''
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.dropna().reset_index(drop=True)
        out_index = []
        for col in df.columns:
            idx = self._3_sigma(df[col])
            out_index.extend(idx)
        out_index = list(set(out_index))
        df = df.drop(out_index, axis=0).reset_index(drop=True)
        return df

    def read_one_csv(self, file_name, nominal_capacity=None):
        '''
        read a csv file and return a DataFrame
        :param file_name: str
        :return: DataFrame with integer 'timestamp' and normalized features
        '''
        df = pd.read_csv(file_name)

        df.insert(df.shape[1], 'timestamp',
            np.arange(df.shape[0], dtype=float))

        # add float mode flag: +1.0 for charge, -1.0 for discharge
        mode_folder = os.path.basename(os.path.dirname(os.path.dirname(file_name)))
        mode_flag = 1.0 if mode_folder == 'charge' else -1.0
        df.insert(df.shape[1], 'mode', float(mode_flag))

        # drop outliers
        df = self.delete_3_sigma(df)

        # normalize all feature columns except 'SoC' and 'SoH'
        feat_cols = [c for c in df.columns if c not in ('SoC', 'SoH')]

        eps = 1e-8

        if self.normalization:
            f_df = df[feat_cols]
            if self.normalization_method == 'min-max':
                denom = (f_df.max() - f_df.min()).replace(0, eps)
                f_df = 2 * (f_df - f_df.min())/denom - 1
            elif self.normalization_method == 'z-score':
                std = f_df.std().replace(0, eps)
                f_df = (f_df - f_df.mean())/std
            df.loc[:, feat_cols] = f_df

        # normalize SoC and SoH by dividing by 100
        if 'SoC' in df.columns:
            df['SoC'] = df['SoC'] / 100.0
        if 'SoH' in df.columns:
            df['SoH'] = df['SoH'] / 100.0


        return df

    def load_one_battery(self, path, nominal_capacity=None):
        '''
        Read a csv file and divide the data into x and y pairs
        :param path: str
        :return: (x1, y1), (x2, y2)
        '''
        df = self.read_one_csv(path, nominal_capacity)
        # SoC
        y = df['SoC'].values
        # all columns except SoC
        feature_cols = [
        'Voltage_measured',
        'Current_measured',
        'Temperature_measured',
        'Current_charge',
        'Voltage_charge',
        'Time',
        'timestamp',
        'mode'
        ]
        x = df[feature_cols].values

        x1 = x[:-1]
        x2 = x[1:]
        y1 = y[:-1]
        y2 = y[1:]
        return (x1, y1), (x2, y2)
    
    def load_all_battery(self, path_list, nominal_capacity=None):
        '''
        Read all csv files, divide the data into X and Y, and then package it into a dataloader
        :param path_list: list of file paths
        :param nominal_capacity: nominal capacity, unused here
        :param batch_size: batch size
        :return: dict of DataLoaders
        '''
        X1, X2, Y1, Y2 = [], [], [], []
        if self.args.log_dir is not None and self.args.save_folder is not None:
            save_name = os.path.join(self.args.save_folder, self.args.log_dir)
            write_to_txt(save_name, 'data path:')
            write_to_txt(save_name, str(path_list))

        for path in path_list:
            (x1, y1), (x2, y2) = self.load_one_battery(path, nominal_capacity)
            X1.append(x1)
            X2.append(x2)
            Y1.append(y1)
            Y2.append(y2)

        X1 = np.concatenate(X1, axis=0)
        X2 = np.concatenate(X2, axis=0)
        Y1 = np.concatenate(Y1, axis=0)
        Y2 = np.concatenate(Y2, axis=0)

        tensor_X1 = torch.from_numpy(X1).float()
        tensor_X2 = torch.from_numpy(X2).float()
        tensor_Y1 = torch.from_numpy(Y1).float().view(-1,1)
        tensor_Y2 = torch.from_numpy(Y2).float().view(-1,1)

        # Condition 1: 80% train+valid, 20% test
        split = int(tensor_X1.shape[0] * 0.8)
        train_X1, test_X1 = tensor_X1[:split], tensor_X1[split:]
        train_X2, test_X2 = tensor_X2[:split], tensor_X2[split:]
        train_Y1, test_Y1 = tensor_Y1[:split], tensor_Y1[split:]
        train_Y2, test_Y2 = tensor_Y2[:split], tensor_Y2[split:]

        train_X1, valid_X1, train_X2, valid_X2, train_Y1, valid_Y1, train_Y2, valid_Y2 = \
            train_test_split(
                train_X1, train_X2, train_Y1, train_Y2,
                test_size=0.2, random_state=420
            )

        train_loader = DataLoader(
            TensorDataset(train_X1, train_X2, train_Y1, train_Y2),
            batch_size=self.args.batch_size, shuffle=False
        )
        valid_loader = DataLoader(
            TensorDataset(valid_X1, valid_X2, valid_Y1, valid_Y2),
            batch_size=self.args.batch_size, shuffle=False
        )
        test_loader = DataLoader(
            TensorDataset(test_X1, test_X2, test_Y1, test_Y2),
            batch_size=self.args.batch_size, shuffle=False
        )

        # Condition 2: single 80/20 split on all data
        train_X1, valid_X1, train_X2, valid_X2, train_Y1, valid_Y1, train_Y2, valid_Y2 = \
            train_test_split(
                tensor_X1, tensor_X2, tensor_Y1, tensor_Y2,
                test_size=0.2, random_state=420
            )
        train_loader_2 = DataLoader(
            TensorDataset(train_X1, train_X2, train_Y1, train_Y2),
            batch_size=self.args.batch_size, shuffle=False
        )
        valid_loader_2 = DataLoader(
            TensorDataset(valid_X1, valid_X2, valid_Y1, valid_Y2),
            batch_size=self.args.batch_size, shuffle=False
        )

        # Condition 3: all data as test
        test_loader_3 = DataLoader(
            TensorDataset(tensor_X1, tensor_X2, tensor_Y1, tensor_Y2),
            batch_size=self.args.batch_size, shuffle=False
        )

        loader = {
            'train': train_loader, 'valid': valid_loader, 'test': test_loader,
            'train_2': train_loader_2, 'valid_2': valid_loader_2, 'test_3': test_loader_3
        }
        return loader


class XJTUdata(DF):
    def __init__(self, root, args):
        super(XJTUdata, self).__init__(args)
        self.root = root
        self.file_list = os.listdir(root)
        self.variables = pd.read_csv(os.path.join(root, self.file_list[0])).columns
        self.num = len(self.file_list)
        self.batch_names = ['2C','3C','R2.5','R3','RW','satellite']
        self.batch_size = args.batch_size

        if self.normalization:
            self.nominal_capacity = 2.0
        else:
            self.nominal_capacity = None
        #print('-'*20,'XJTU data','-'*20)

    def read_one_batch(self,batch='2C'):
        '''
        Read a batch of csv files, divide the data into four parts: x1, y1, x2, y2, and encapsulate it into a dataloader
        :param batch: int or str:batch
        :return: dict
        '''
        if isinstance(batch,int):
            batch = self.batch_names[batch]
        assert batch in self.batch_names, 'batch must be in {}'.format(self.batch_names)
        file_list = []
        for i in range(self.num):
            if batch in self.file_list[i]:
                path = os.path.join(self.root,self.file_list[i])
                file_list.append(path)
        return self.load_all_battery(path_list=file_list,nominal_capacity=self.nominal_capacity)

    def read_all(self,specific_path_list=None):
        '''
        Read all csv files, divide the data into four parts: x1, y1, x2, y2, and encapsulate it into a dataloader
        :return: dict
        '''
        if specific_path_list is None:
            file_list = []
            for file in self.file_list:
                path = os.path.join(self.root, file)
                file_list.append(path)
            return self.load_all_battery(path_list=file_list,nominal_capacity=self.nominal_capacity)
        else:
            return self.load_all_battery(path_list=specific_path_list,nominal_capacity=self.nominal_capacity)


class HUSTdata(DF):
    def __init__(self,root='../data/HUST data',args=None):
        super(HUSTdata, self).__init__(args)
        self.root = root
        if self.normalization:
            self.nominal_capacity = 1.1
        else:
            self.nominal_capacity = None
        #print('-'*20,'HUST data','-'*20)

    def read_all(self,specific_path_list=None):
        '''
        Read all csv files.
        If specific_path_list is not None, read the specified file;
        otherwise read all files;
        :param self:
        :param specific_path:
        :return: dict
        '''
        if specific_path_list is None:
            file_list = []
            files = os.listdir(self.root)
            for file in files:
                path = os.path.join(self.root,file)
                file_list.append(path)
            return self.load_all_battery(path_list=file_list, nominal_capacity=self.nominal_capacity)
        else:
            return self.load_all_battery(path_list=specific_path_list, nominal_capacity=self.nominal_capacity)


class MITdata(DF):
    def __init__(self,root='../data/MIT data',args=None):
        super(MITdata, self).__init__(args)
        self.root = root
        self.batchs = ['2017-05-12','2017-06-30','2018-04-12']
        if self.normalization:
            self.nominal_capacity = 1.1
        else:
            self.nominal_capacity = None
        #print('-' * 20, 'MIT data', '-' * 20)

    def read_one_batch(self,batch):
        '''
        Read a batch of csv files
        :param batch: int,可选[1,2,3]
        :return: dict
        '''
        assert batch in [1,2,3], 'batch must be in {}'.format([1,2,3])
        root = os.path.join(self.root,self.batchs[batch-1])
        file_list = os.listdir(root)
        path_list = []
        for file in file_list:
            file_name = os.path.join(root,file)
            path_list.append(file_name)
        return self.load_all_battery(path_list=path_list, nominal_capacity=self.nominal_capacity)

    def read_all(self,specific_path_list=None):
        '''
        Read all csv files.
        If specific_path_list is not None, read the specified file; otherwise read all files;
        :param self:
        :return: dict
        '''
        if specific_path_list is None:
            file_list = []
            for batch in self.batchs:
                root = os.path.join(self.root,batch)
                files = os.listdir(root)
                for file in files:
                    path = os.path.join(root,file)
                    file_list.append(path)
            return self.load_all_battery(path_list=file_list, nominal_capacity=self.nominal_capacity)
        else:
            return self.load_all_battery(path_list=specific_path_list, nominal_capacity=self.nominal_capacity)

class TJUdata(DF):
    def __init__(self,root='../data/TJU data',args=None):
        super(TJUdata, self).__init__(args)
        self.root = root
        self.batchs = ['Dataset_1_NCA_battery','Dataset_2_NCM_battery','Dataset_3_NCM_NCA_battery']
        if self.normalization:
            self.nominal_capacities = [3.5,3.5,2.5]
        else:
            self.nominal_capacities = [None,None,None]
        #print('-' * 20, 'TJU data', '-' * 20)

    def read_one_batch(self,batch):
        '''
        Read a batch of csv files
        :param batch: int,可选[1,2,3]; optional[1,2,3]
        :return: DataFrame
        '''
        assert batch in [1,2,3], 'batch must be in {}'.format([1,2,3])
        root = os.path.join(self.root,self.batchs[batch-1])
        file_list = os.listdir(root)
        df = pd.DataFrame()
        path_list = []
        for file in file_list:
            file_name = os.path.join(root,file)
            path_list.append(file_name)
        return self.load_all_battery(path_list=path_list, nominal_capacity=self.nominal_capacities[batch])

    def read_all(self,specific_path_list):
        '''
        Read all csv files and encapsulate them into a dataloader
        :param self:
        :return: dict
        '''
        for i,batch in enumerate(self.batchs):
            if batch in specific_path_list[0]:
                normal_capacity = self.nominal_capacities[i]
                break
        return self.load_all_battery(path_list=specific_path_list, nominal_capacity=normal_capacity)

class NASAdata(DF):
    def __init__(self, root='../data/NASA data', args=None):

        super(NASAdata, self).__init__(args)
        self.root       = root
        self.modes      = ['charge', 'discharge']
        # discover battery IDs under each mode
        self.battery_ids = {
            mode: sorted(os.listdir(os.path.join(root, mode)))
            for mode in self.modes
        }
        self.batch_size = args.batch_size
        # nominal_capacity no longer used by DF.read_one_csv,
        # but we keep it here for API compatibility:
        self.nominal_capacity = None

    def read_one_batch(self, mode, batch):
        """
        Read just a single battery's data in one mode.
        :param mode:    'charge' or 'discharge'
        :param batch:   either an int index into self.battery_ids[mode] or the battery ID str
        :return:        dict of DataLoaders (same API as load_all_battery)
        """
        assert mode in self.modes, f"mode must be one of {self.modes}"
        # allow integer or string batch selection
        if isinstance(batch, int):
            batch = self.battery_ids[mode][batch]
        assert batch in self.battery_ids[mode], f"{batch} not found under {mode}"
        folder = os.path.join(self.root, mode, batch)
        # collect all CSV files in that folder
        paths = [
            os.path.join(folder, fn)
            for fn in os.listdir(folder)
            if fn.lower().endswith('.csv')
        ]
        return self.load_all_battery(path_list=paths,
                                     nominal_capacity=self.nominal_capacity)
    
    def read_all(self, mode=None):
        modes = [mode] if mode in self.modes else self.modes
        paths = []
        for m in modes:
            for batch in self.battery_ids[m]:
                folder = os.path.join(self.root, m, batch)
                for fn in os.listdir(folder):
                    if not fn.lower().endswith('.csv'):
                        continue
                    full = os.path.join(folder, fn)
                    # SKIP any empty files
                    if os.path.getsize(full) == 0:
                        print(f"Skipping empty file {full}")
                        continue
                    paths.append(full)

        return self.load_all_battery(path_list=paths,
                                     nominal_capacity=self.nominal_capacity)
    
    def read_specific(self, file_list):
        """
        Read exactly the CSV files in `file_list` (absolute paths or paths
        relative to self.root), then package into the usual train/valid/test
        DataLoader dict via load_all_battery.
        
        :param file_list: list of str paths (absolute or relative to self.root)
        :return:          dict of DataLoaders, same API as read_all/load_one_batch
        """
        import os

        # Resolve each entry into a full path under self.root if needed
        resolved = []
        for p in file_list:
            full = p if os.path.isabs(p) else os.path.join(self.root, p)
            if not os.path.isfile(full):
                raise FileNotFoundError(f"{full} does not exist")
            if not full.lower().endswith('.csv'):
                raise ValueError(f"{full} is not a .csv file")
            if os.path.getsize(full) == 0:
                print(f"Skipping empty file {full}")
                continue
            resolved.append(full)

        if not resolved:
            raise ValueError("No valid CSV files to read in read_specific()")

        # Delegate to your existing loader
        return self.load_all_battery(path_list=resolved,
                                     nominal_capacity=self.nominal_capacity)
    
if __name__ == '__main__':
    import argparse
    def get_args():
        parser = argparse.ArgumentParser()
        parser.add_argument('--data',type=str,default='MIT',help='XJTU, HUST, MIT, TJU')
        parser.add_argument('--batch',type=int,default=1,help='1,2,3')
        parser.add_argument('--batch_size',type=int,default=256,help='batch size')
        parser.add_argument('--normalization_method',type=str,default='z-score',help='min-max,z-score')
        parser.add_argument('--log_dir',type=str,default='test.txt',help='log dir')
        return parser.parse_args()

    args = get_args()

    # xjtu = XJTUdata(root='../data/XJTU data',args=args)
    # path = '../data/XJTU data/2C_battery-1.csv'
    # xjtu.read_one_batch('2C')
    # xjtu.read_all()
    #
    # hust = HUSTdata(args=args)
    # hust.read_all()
    #
    mit = MITdata(args=args)
    mit.read_one_batch(batch=1)
    loader = mit.read_all()

    # tju = HUSTdata(args=args)
    # loader = tju.read_all()
    train_loader = loader['train']
    test_loader = loader['test']
    valid_loader = loader['valid']
    all_loader = loader['test_3']
    print('train_loader:',len(train_loader),'test_loader:',len(test_loader),'valid_loader:',len(valid_loader),'all_loader:',len(all_loader))

    for iter,(x1,x2,y1,y2) in enumerate(train_loader):
        print('x1 shape:',x1.shape)
        print('x2 shape:',x2.shape)
        print('y1 shape:',y1.shape)
        print('y2 shape:',y2.shape)
        print('y1 max:',y1.max())
        break








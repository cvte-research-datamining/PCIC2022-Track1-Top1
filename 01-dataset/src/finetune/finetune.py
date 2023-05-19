import numpy as np
import pandas as pd
import os
import gc
import random
from tqdm import tqdm
from joblib import Parallel, delayed
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score
import torch
from torch import nn
from torch.optim import Adam, lr_scheduler
from torch.utils.data import DataLoader, TensorDataset
from torch.cuda.amp import autocast, GradScaler


class RNN(nn.Module):
    def __init__(self, input_size, hidden_sizes, seq_len, dropout=0.5, output_size=1):
        super(RNN, self).__init__()
        self.lstm_1 = nn.LSTM(input_size, hidden_sizes[0], num_layers=2, batch_first=True, bidirectional=True, dropout=dropout)
        self.lstm_21 = nn.LSTM(2 * hidden_sizes[0], hidden_sizes[1], num_layers=2, batch_first=True, bidirectional=True, dropout=dropout)
        self.lstm_22 = nn.LSTM(input_size, hidden_sizes[1], num_layers=2, batch_first=True, bidirectional=True, dropout=dropout)
        self.lstm_31 = nn.LSTM(2 * hidden_sizes[1], hidden_sizes[2], num_layers=2, batch_first=True, bidirectional=True, dropout=dropout)
        self.lstm_32 = nn.LSTM(4 * hidden_sizes[1], hidden_sizes[2], num_layers=2, batch_first=True, bidirectional=True, dropout=dropout)
        self.lstm_41 = nn.LSTM(2 * hidden_sizes[2], hidden_sizes[3], num_layers=2, batch_first=True, bidirectional=True, dropout=dropout)
        self.lstm_42 = nn.LSTM(4 * hidden_sizes[2], hidden_sizes[3], num_layers=2, batch_first=True, bidirectional=True, dropout=dropout)
        hidd = 2 * hidden_sizes[0] + 4 * (hidden_sizes[1] + hidden_sizes[2] + hidden_sizes[3])
        self.lstm_5 = nn.LSTM(hidd, hidden_sizes[4], num_layers=2, batch_first=True, bidirectional=True, dropout=dropout)
        
        self.fc = nn.Sequential(
            nn.Linear(2 * hidden_sizes[4] * seq_len, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(4096, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(1024, output_size)
        )
        
    def forward(self, x):
        x1, _ = self.lstm_1(x)

        x_x1, _ = self.lstm_21(x1)
        x_x2, _ = self.lstm_22(x)
        x2 = torch.cat([x_x1, x_x2], dim=2)
        
        x_x1, _ = self.lstm_31(x_x1)
        x_x2, _ = self.lstm_32(x2)
        x3 = torch.cat([x_x1, x_x2], dim=2)
        
        x_x1, _ = self.lstm_41(x_x1)
        x_x2, _ = self.lstm_42(x3)
        x4 = torch.cat([x_x1, x_x2], dim=2)
        
        x = torch.cat([x1, x2, x3, x4], dim=2)
        x, _ = self.lstm_5(x)

        # fully connected layers:
        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)
        return x


def add_features(df, features):
    for feature in features:
        df_grouped = df.groupby(['sequence'])[feature]
        df[feature + '_lag1'] = df_grouped.shift(1)
        df[feature + '_lag1'].fillna(df[feature].median(), inplace=True)
        df[feature + '_diff1'] = df[feature] - df[feature + '_lag1']

        df[feature + '_lag2'] = df_grouped.shift(2)
        df[feature + '_lag2'].fillna(df[feature].median(), inplace=True)
        df[feature + '_diff2'] = df[feature] - df[feature + '_lag2']
        
        df_rolling = df_grouped.rolling(10, center=True)
        df[feature + '_roll_mean'] = df_rolling.mean().reset_index(0, drop=True)
        df[feature + '_roll_std'] = df_rolling.std().reset_index(0, drop=True)
    df.dropna(axis=0, inplace=True)
    return df


def get_feat(df, features):
    sequences = df['sequence'].unique().tolist()
    sequences_gp = []
    for i in range(0, len(sequences), 500):
        sequences_gp.append(df[df['sequence'].isin(sequences[i: i+500])])
    res = Parallel(backend='multiprocessing', n_jobs=16)(delayed(add_features)(df_tmp, features) for df_tmp in tqdm(sequences_gp))
    df = pd.concat(res, ignore_index=True)
    return df


def load_data(load_path):
    train = pd.read_feather(load_path + 'df_a.feather')
    train_labels = pd.read_feather(load_path + 'df_a_y.feather')
    test = pd.read_feather(load_path + 'df_b_test.feather')
    train_b = pd.read_feather(load_path + 'df_b.feather')
    train_labels_b = pd.read_feather(load_path + 'df_b_y.feather')

    train_b['sequence'] = train_b['sequence'] + train['sequence'].max() + 1
    train_labels_b['sequence'] = train_labels_b['sequence'] + train_labels['sequence'].max() + 1

    # train = pd.concat([train, train_b], ignore_index=True)
    # train_labels = pd.concat([train_labels, train_labels_b], ignore_index=True)
    return train, test, train_b, train_labels, train_labels_b


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def main():
    seed = 601
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    DEBUG = False
    batch_size = 128
    hidden_sizes = [288, 192, 144, 96, 32]
    lr = 0.0001
    epochs = 5
    n_splits = 10
    warm_up = 0.0
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    load_path = '../input/'
    to_path = './'

    train, test, train_b, train_labels, train_labels_b = load_data(load_path)
    del train, train_labels
    gc.collect()

    if DEBUG:
        epochs = 2
        n_splits = 2
        train_b = train_b.head(672 * 500)
        test = test.head(672 * 500)
        train_labels_b = train_labels_b.head(500)

    features = ['sensor_{}'.format(i) for i in range(10)]

    train_b = get_feat(train_b, features)
    print(f'train_b: {train_b.shape}')

    test = get_feat(test, features)
    print(f'test: {test.shape}')

    train_b = train_b.set_index(['sequence', 'step'])
    test = test.set_index(['sequence', 'step'])

    input_size = train_b.shape[1]
    sequence_length = len(train_b.index.get_level_values(1).unique())

    # Scaling test and train
    scaler = StandardScaler()
    train = scaler.fit_transform(train_b)
    # train_b = scaler.fit_transform(train_b)
    test = scaler.transform(test)

    # train = np.concatenate((train, train_b), axis=0)
    # train_labels = pd.concat([train_labels, train_labels_b], ignore_index=True)
    # print('concat train shape: ', train.shape)
    # print('concat train_labels shape: ', train_labels.shape)

    # Reshaping:
    train = train.reshape(-1, sequence_length, input_size)
    test = test.reshape(-1, sequence_length, input_size)
    print('After Reshape')
    print('Shape of train set: {}'.format(train.shape))
    print('Shape of test set: {}'.format(test.shape))

    test_tensor = torch.tensor(test).float()

    labels_np = train_labels_b['label'].values

    f1s = []
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for fold, (train_index, test_index) in enumerate(kf.split(train)):
        t_X, v_X = train[train_index], train[test_index]
        t_y, v_y = labels_np[train_index], labels_np[test_index]
        
        valid_label1_count = np.sum(v_y == 1)
        print(f'\nFold={fold + 1}, valid label 1 count: {valid_label1_count}')

        train_X_tensor = torch.tensor(t_X).float()
        val_X_tensor = torch.tensor(v_X).float()

        train_y_tensor = torch.tensor(t_y)
        val_y_tensor = torch.tensor(v_y)

        train_tensor = TensorDataset(train_X_tensor, train_y_tensor)
        val_tensor = TensorDataset(val_X_tensor, val_y_tensor)

        # Defining the dataloaders
        dataloaders = dict()
        dataloaders['train'] = DataLoader(train_tensor, batch_size=batch_size, shuffle=True)
        dataloaders['val'] = DataLoader(val_tensor, batch_size=batch_size)
        dataloaders['test'] = DataLoader(test_tensor, batch_size=batch_size)

        model = RNN(input_size, hidden_sizes, sequence_length)
        model.load_state_dict(torch.load(f'best-state-3.pt', map_location='cpu'))
        model.to(device)
        optimizer = Adam(model.parameters(), lr=lr)
        scheduler = lr_scheduler.OneCycleLR(optimizer, max_lr=lr, epochs=epochs, steps_per_epoch=len(dataloaders['train']),
                                            pct_start=0.2, anneal_strategy='cos')
        
        criterion = nn.BCEWithLogitsLoss()
        best_score = -999.0
        best_epoch = 0
        scaler = GradScaler()
        for epoch in range(epochs):
            train_loss = AverageMeter()
            model.train()
            for batch_x, labels in tqdm(dataloaders['train']):
                batch_x, labels = batch_x.to(device), labels.to(device)
                labels = labels.unsqueeze(1).float()
                with autocast():
                    output = model.forward(batch_x)
                    loss = criterion(output, labels)
                scaler.scale(loss).backward()
                train_loss.update(loss.item(), output.size(0))
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()

                del batch_x, labels
                gc.collect()
                torch.cuda.empty_cache()

            valid_loss = AverageMeter()
            model.eval()
            preds_all = torch.LongTensor()
            labels_all = torch.LongTensor()
            for batch_x, labels in tqdm(dataloaders['val']):
                labels_all = torch.cat((labels_all, labels), dim=0)
                batch_x, labels = batch_x.to(device), labels.to(device)
                labels = labels.unsqueeze(1).float()
                with torch.no_grad():
                    output = model.forward(batch_x)
                    loss = criterion(output, labels)
                preds_all = torch.cat((preds_all, torch.sigmoid(output).to('cpu')), dim=0)
                valid_loss.update(loss.item(), output.size(0))
                batch_x = batch_x.cpu()
                labels = labels.cpu()
                del batch_x, labels
                gc.collect()
                torch.cuda.empty_cache()
            
            df_preds_all = pd.DataFrame({'prob': preds_all[:, 0]})
            df_preds_all['rank'] = df_preds_all['prob'].rank()
            preds_all_threshold = np.where(df_preds_all['rank'] > (len(df_preds_all) - valid_label1_count), 1, 0)
            current_score = f1_score(labels_all, preds_all_threshold)
            
            if current_score > best_score:
                best_score = current_score
                best_epoch = epoch
                torch.save(model.state_dict(), to_path + f'best-state-finetune-{fold+1}.pt')
            to_print = 'Epoch: ' + str(epoch + 1) + ' of ' + str(epochs)
            to_print += ' - Train Loss: {:.4f}'.format(train_loss.avg)
            to_print += ' - Valid Loss: {:.4f}'.format(valid_loss.avg)
            to_print += ' - Valid F1: {:.4f}'.format(current_score)
            print(to_print)
        
        model = model.cpu()
        del model
        gc.collect()
        torch.cuda.empty_cache()
        f1s.append(best_score)
        to_print = f'\nFold={fold+1} Best Valid F1 is: {best_score:.4f} after {best_epoch + 1} epochs'
        print(to_print)
    print(f'\n f1 scores: {f1s}, f1s mean={np.mean(f1s)}')

    for i in range(n_splits):
        model = RNN(input_size, hidden_sizes, sequence_length)
        model.load_state_dict(torch.load(to_path + f'best-state-finetune-{i+1}.pt', map_location='cpu'))
        model.to(device)
        model.eval()
        preds_all = torch.LongTensor()
        with torch.no_grad():
            for batch_x in tqdm(dataloaders['test']):
                batch_x = batch_x.to(device)
                output = model.forward(batch_x)
                preds_all = torch.cat((preds_all, torch.sigmoid(output).to('cpu')), dim=0)
        if i == 0:
            df_preds_all = pd.DataFrame({f'prob{i+1}': preds_all[:, 0]})
        else:
            df_preds_all[f'prob{i+1}'] = preds_all[:, 0]
        
        model = model.cpu()
        del model
        torch.cuda.empty_cache()
    
    df_preds_all['prob_mean'] = df_preds_all.apply(lambda x: x.mean(), axis=1)
    df_preds_all.to_csv(to_path + 'prob.csv', index=False)
    
    df_preds_all['rank'] = df_preds_all['prob_mean'].rank()
    preds_all_threshold = np.where(df_preds_all['rank'] > (len(df_preds_all) - 6387), 1, 0)
    
    sub = pd.DataFrame({'label': preds_all_threshold})
    sub[['label']].to_csv(to_path + 'submission.csv', index=False, header=False)
    print('Resuls are saved to submission.csv')


if __name__ == '__main__':
    main()

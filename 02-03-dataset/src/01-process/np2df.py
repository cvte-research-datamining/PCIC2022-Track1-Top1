import numpy as np
import pandas as pd
import os
from tqdm import tqdm


def np2df(path, base_path='./output/'):
    path_split = path.split('/')
    if 'train' in path or 'test' in path:
        data_path = path_split[-4] + '_' + path_split[-3] + '_' + path_split[-2]
    else:
        data_path = path_split[-3] + '_' + path_split[-2]
    X_a = np.load(path + 'X.npy')
    if 'test' not in path:
        y_a = np.load(path + 'Y.npy')
        print(f'{data_path} origin shape: {X_a.shape} - {y_a.shape}')
    else:
        print(f'{data_path} origin shape: {X_a.shape}')
    
    a_list = []
    for i in tqdm(range(len(X_a))):
        df_tmp = pd.DataFrame(X_a[i])
        df_tmp.columns = [f'sensor_{i}' for i in range(10)]
        df_tmp['sequence'] = i
        df_tmp['step'] = range(672)
        a_list.append(df_tmp)
    df_a = pd.concat(a_list, ignore_index=True)
    df_a = df_a[['sequence', 'step'] + [f'sensor_{i}' for i in range(10)]]
    df_a.to_feather(base_path + f'df_{data_path}.feather')

    if 'test' not in path:
        df_a_y = pd.DataFrame(y_a)
        df_a_y.columns = ['label']
        df_a_y['sequence'] = range(len(y_a))
        df_a_y = df_a_y[['sequence', 'label']]
        df_a_y.to_feather(base_path + f'df_{data_path}_y.feather')
        print(f'{data_path} df shape:  {df_a.shape} - {df_a_y.shape}')
    else:
        print(f'{data_path} df shape:  {df_a.shape}')


if __name__ == '__main__':
    base_path = './output/'
    if not os.path.exists(base_path):
        os.makedirs(base_path)

    path_pre = '../../../../dataset/'
    np2df(path_pre + 'dataset2/cityA/', base_path)
    np2df(path_pre + 'dataset2/cityB/train/', base_path)
    np2df(path_pre + 'dataset2/cityB/test/', base_path)

    np2df(path_pre + 'dataset3/cityA/', base_path)
    np2df(path_pre + 'dataset3/cityB/train/', base_path)
    np2df(path_pre + 'dataset3/cityB/test/', base_path)

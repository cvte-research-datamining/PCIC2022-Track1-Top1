import numpy as np
import pandas as pd
import os
import gc
from tqdm import tqdm


def load_data():
    path = '../../01-process/output/'
    train = pd.read_feather(path + 'df_dataset3_cityA.feather')
    train_labels = pd.read_feather(path + 'df_dataset3_cityA_y.feather')

    train_b = pd.read_feather(path + 'df_dataset3_cityB_train.feather')
    train_labels_b = pd.read_feather(path + 'df_dataset3_cityB_train_y.feather')
    
    test = pd.read_feather(path + 'df_dataset3_cityB_test.feather')

    train_b['sequence'] = train_b['sequence'] + train['sequence'].max() + 1
    train_labels_b['sequence'] = train_labels_b['sequence'] + train_labels['sequence'].max() + 1

    test['sequence'] = test['sequence'] + train_b['sequence'].max() + 1

    return train, test, train_b, train_labels, train_labels_b


def sensor_process(df_train):
    for i in tqdm(range(10)):
        if i not in [3, 7]:
            df_sensor = df_train.groupby(['step'])[[f'sensor_{i}']].mean()
            df_sensor.rename(columns={f'sensor_{i}': f'sensor_{i}_mean'}, inplace=True)
            df_train = df_train.merge(df_sensor, left_on=['step'], right_index=True)
            df_train[f'sensor_{i}_diff_mean'] = df_train[f'sensor_{i}'] - df_train[f'sensor_{i}_mean']
            df_train[f'sensor_{i}_diff_mean_large_0'] = np.where(df_train[f'sensor_{i}_diff_mean'] > 0, 1, 0)
            del df_sensor
            gc.collect()

            # label 1
            df_train_label1 = df_train[df_train['label'] == 1]

            df_sensor_label1 = df_train_label1.groupby(['step'])[[f'sensor_{i}']].mean()
            df_sensor_label1.rename(columns={f'sensor_{i}': f'sensor_{i}_mean_label1'}, inplace=True)

            df_train = df_train.merge(df_sensor_label1, left_on=['step'], right_index=True)
            df_train[f'sensor_{i}_diff_mean_label1'] = df_train[f'sensor_{i}'] - df_train[f'sensor_{i}_mean_label1']
            df_train[f'sensor_{i}_diff_mean_label1_large_0'] = np.where(df_train[f'sensor_{i}_mean_label1'] > 0, 1, 0)

            # label 0
            df_train_label0 = df_train[df_train['label'] == 0]

            df_train_label0 = df_train_label0.groupby(['step'])[[f'sensor_{i}']].mean()
            df_train_label0.rename(columns={f'sensor_{i}': f'sensor_{i}_mean_label0'}, inplace=True)

            df_train = df_train.merge(df_train_label0, left_on=['step'], right_index=True)
            df_train[f'sensor_{i}_diff_mean_label0'] = df_train[f'sensor_{i}'] - df_train[f'sensor_{i}_mean_label0']
            df_train[f'sensor_{i}_diff_mean_label0_large_0'] = np.where(df_train[f'sensor_{i}_mean_label0'] > 0, 1, 0)

    df_train.drop('label', axis=1, inplace=True)
    gc.collect()
    df_train.sort_values(['sequence', 'step'], inplace=True)
    return df_train


def sensor_process_cityb(df_train_b, df_test):
    for i in tqdm(range(10)):
        if i not in [3, 7]:
            df_sensor = df_train_b.groupby(['step'])[[f'sensor_{i}']].mean()
            df_sensor.rename(columns={f'sensor_{i}': f'sensor_{i}_mean'}, inplace=True)
            df_train_b = df_train_b.merge(df_sensor, left_on=['step'], right_index=True)
            df_train_b[f'sensor_{i}_diff_mean'] = df_train_b[f'sensor_{i}'] - df_train_b[f'sensor_{i}_mean']
            df_train_b[f'sensor_{i}_diff_mean_large_0'] = np.where(df_train_b[f'sensor_{i}_diff_mean'] > 0, 1, 0)
            del df_sensor
            gc.collect()

            df_sensor = df_test.groupby(['step'])[[f'sensor_{i}']].mean()
            df_sensor.rename(columns={f'sensor_{i}': f'sensor_{i}_mean'}, inplace=True)
            df_test = df_test.merge(df_sensor, left_on=['step'], right_index=True)
            df_test[f'sensor_{i}_diff_mean'] = df_test[f'sensor_{i}'] - df_test[f'sensor_{i}_mean']
            df_test[f'sensor_{i}_diff_mean_large_0'] = np.where(df_test[f'sensor_{i}_diff_mean'] > 0, 1, 0)
            del df_sensor
            gc.collect()
            
            # label 1
            df_train_b_label1 = df_train_b[df_train_b['label'] == 1]

            df_sensor_label1 = df_train_b_label1.groupby(['step'])[[f'sensor_{i}']].mean()
            df_sensor_label1.rename(columns={f'sensor_{i}': f'sensor_{i}_mean_label1'}, inplace=True)

            df_train_b = df_train_b.merge(df_sensor_label1, left_on=['step'], right_index=True)
            df_train_b[f'sensor_{i}_diff_mean_label1'] = df_train_b[f'sensor_{i}'] - df_train_b[f'sensor_{i}_mean_label1']
            df_train_b[f'sensor_{i}_diff_mean_label1_large_0'] = np.where(df_train_b[f'sensor_{i}_mean_label1'] > 0, 1, 0)

            df_test = df_test.merge(df_sensor_label1, left_on=['step'], right_index=True)
            df_test[f'sensor_{i}_diff_mean_label1'] = df_test[f'sensor_{i}'] - df_test[f'sensor_{i}_mean_label1']
            df_test[f'sensor_{i}_diff_mean_label1_large_0'] = np.where(df_test[f'sensor_{i}_mean_label1'] > 0, 1, 0)

            # label 0
            df_train_b_label0 = df_train_b[df_train_b['label'] == 0]

            df_train_b_label0 = df_train_b_label0.groupby(['step'])[[f'sensor_{i}']].mean()
            df_train_b_label0.rename(columns={f'sensor_{i}': f'sensor_{i}_mean_label0'}, inplace=True)

            df_train_b = df_train_b.merge(df_train_b_label0, left_on=['step'], right_index=True)
            df_train_b[f'sensor_{i}_diff_mean_label0'] = df_train_b[f'sensor_{i}'] - df_train_b[f'sensor_{i}_mean_label0']
            df_train_b[f'sensor_{i}_diff_mean_label0_large_0'] = np.where(df_train_b[f'sensor_{i}_mean_label0'] > 0, 1, 0)

            df_test = df_test.merge(df_train_b_label0, left_on=['step'], right_index=True)
            df_test[f'sensor_{i}_diff_mean_label0'] = df_test[f'sensor_{i}'] - df_test[f'sensor_{i}_mean_label0']
            df_test[f'sensor_{i}_diff_mean_label0_large_0'] = np.where(df_test[f'sensor_{i}_mean_label0'] > 0, 1, 0)

    df_train_b.drop('label', axis=1, inplace=True)
    gc.collect()
    df_train_b.sort_values(['sequence', 'step'], inplace=True)
    df_test.sort_values(['sequence', 'step'], inplace=True)
    return df_train_b, df_test


def main():
    DEBUG = False
    to_path = './output/'
    if not os.path.exists(to_path):
        os.makedirs(to_path)

    train, test, train_b, train_labels, train_labels_b = load_data()
    
    train_labels.to_feather(to_path + 'train_labels.feather')
    train_labels_b.to_feather(to_path + 'train_labels_b.feather')

    if DEBUG:
        train = train.head(672 * 500)
        train_b = train_b.head(672 * 500)
        test = test.head(672 * 500)
        train_labels = train_labels.head(500)
        train_labels_b = train_labels_b.head(500)

    train = train.merge(train_labels, on='sequence', how='left')
    train = sensor_process(train)
    train.reset_index(drop=True, inplace=True)
    print('train shape: ', train.shape)
    train.to_feather(to_path + 'train.feather')
    del train, train_labels
    gc.collect()
    
    train_b = train_b.merge(train_labels_b, on='sequence', how='left')
    train_b, test = sensor_process_cityb(train_b, test)
    train_b.reset_index(drop=True, inplace=True)
    print('train_b shape: ', train_b.shape)
    train_b.to_feather(to_path + 'train_b.feather')
    del train_b, train_labels_b
    gc.collect()

    test.reset_index(drop=True, inplace=True)
    print('test shape: ', test.shape)
    test.to_feather(to_path + 'test.feather')
    del test
    gc.collect()
    
    print('Done!')


if __name__ == '__main__':
    main()

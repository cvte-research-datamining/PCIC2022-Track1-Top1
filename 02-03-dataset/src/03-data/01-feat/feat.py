import numpy as np
import pandas as pd
import gc
import os
from tqdm import tqdm
from joblib import Parallel, delayed
import warnings
warnings.filterwarnings('ignore')


def load_data(path):
    train = pd.read_feather(path + 'train.feather')
    train_b = pd.read_feather(path + 'train_b.feather')
    test = pd.read_feather(path + 'test.feather')
    train_labels = pd.read_feather(path + 'train_labels.feather')
    train_labels_b = pd.read_feather(path + 'train_labels_b.feather')
    return train, train_b, test, train_labels, train_labels_b


def add_features(df, features):
    for feature in features:
        df_grouped = df.groupby(['sequence'])[feature]
        df[feature + '_lag1'] = df_grouped.shift(1)
        df[feature + '_lag1'].fillna(df[feature].median(), inplace=True)
        df[feature + '_diff1'] = df[feature] - df[feature + '_lag1']

        df[feature + '_lag2'] = df_grouped.shift(2)
        df[feature + '_lag2'].fillna(df[feature].median(), inplace=True)
        df[feature + '_diff2'] = df[feature] - df[feature + '_lag2']
        
        # df_rolling = df_grouped.rolling(10, center=True)
        # df[feature + '_roll_mean'] = df_rolling.mean().reset_index(0, drop=True)
        # df[feature + '_roll_std'] = df_rolling.std().reset_index(0, drop=True)
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


def get_col_stats(df, features):
    data = df.drop('step', axis=1)
    del df
    gc.collect()
    gp = data.groupby(['sequence'], as_index=False)[features]

    gp_max = gp.max()
    gp_max.columns = ['sequence'] + [f'sensor_{i}_max' for i in range(10)]

    gp_min = gp.min()
    gp_min.columns = ['sequence'] + [f'sensor_{i}_min' for i in range(10)]

    gp_sum = gp.mean()
    gp_sum.columns = ['sequence'] + [f'sensor_{i}_mean' for i in range(10)]

    gp_var = gp.var()
    gp_var.columns = ['sequence'] + [f'sensor_{i}_var' for i in range(10)]

    df = gp_max.merge(gp_min, on=['sequence'], how='left')
    df = df.merge(gp_sum, on=['sequence'], how='left')
    df = df.merge(gp_var, on=['sequence'], how='left')
    return df


def get_col_stats_parallel(df, features):
    sequences = df['sequence'].unique().tolist()
    sequences_gp = []
    for i in range(0, len(sequences), 500):
        sequences_gp.append(df[df['sequence'].isin(sequences[i: i+500])])
    res = Parallel(backend='multiprocessing', n_jobs=32)(delayed(get_col_stats)(df_tmp, features) for df_tmp in tqdm(sequences_gp))
    df = pd.concat(res, ignore_index=True)
    return df


def get_sensor_t(name, g, i):
    name = pd.DataFrame([name]).T
    name.index = range(len(name))
    gp = pd.DataFrame(np.array(g.drop(['sequence'], axis=1).T).reshape((1, -1)))
    gp.index = range(len(gp))
    res = pd.concat([name, gp], axis=1)
    res.columns = ['sequence'] + \
        [f'sensor_{i}_mean_step_{j}' for j in range(672)] + [f'sensor_{i}_diff_mean_step_{j}' for j in range(672)] + [f'sensor_{i}_diff_mean_large_0_step_{j}' for j in range(672)] + \
        [f'sensor_{i}_mean_label1_step_{j}' for j in range(672)] + [f'sensor_{i}_diff_mean_label1_step_{j}' for j in range(672)] + [f'sensor_{i}_diff_mean_label1_large_0_step_{j}' for j in range(672)] + \
        [f'sensor_{i}_mean_label0_step_{j}' for j in range(672)] + [f'sensor_{i}_diff_mean_label0_step_{j}' for j in range(672)] + [f'sensor_{i}_diff_mean_label0_large_0_step_{j}' for j in range(672)]
    return res


def get_sensor_t_parallel(df):
    for i in tqdm(range(10)):
        if i not in [3, 7]:
            cols = [f'sensor_{i}_mean', f'sensor_{i}_diff_mean', f'sensor_{i}_diff_mean_large_0',
                    f'sensor_{i}_mean_label1', f'sensor_{i}_diff_mean_label1', f'sensor_{i}_diff_mean_label1_large_0',
                    f'sensor_{i}_mean_label0', f'sensor_{i}_diff_mean_label0', f'sensor_{i}_diff_mean_label0_large_0']
            gp = df[['sequence'] + cols].groupby(['sequence'])
            res = Parallel(n_jobs=32)(delayed(get_sensor_t)(name, g, i) for name, g in gp)
            if i == 0:
                df_sensor_t = pd.concat(res)
            else:
                df_sensor_t_tmp = pd.concat(res)
                df_sensor_t.merge(df_sensor_t_tmp, on='sequence', how='left')
                del df_sensor_t_tmp, res, gp
                gc.collect()
    return df_sensor_t


def get_row_t(name, g):
    name = pd.DataFrame([name]).T
    name.index = range(len(name))
    gp = pd.DataFrame(np.array(g.drop(['sequence'], axis=1).T).reshape((1, -1)))
    gp.index = range(len(gp))
    res = pd.concat([name, gp], axis=1)
    res.columns = ['sequence'] + ['row_max{}'.format(i) for i in range(672)] + ['row_min{}'.format(i) for i in range(672)] + ['row_mean{}'.format(i) for i in range(672)] + ['row_var{}'.format(i) for i in range(672)]
    return res


def get_row_t_parallel(df):
    df['row_max'] = df.loc[:, 'sensor_0':'sensor_9'].max(1)
    df['row_min'] = df.loc[:, 'sensor_0':'sensor_9'].min(1)
    df['row_mean'] = df.loc[:, 'sensor_0':'sensor_9'].mean(1)
    df['row_var'] = df.loc[:, 'sensor_0':'sensor_9'].var(1)
    
    gp = df[['sequence', 'row_max', 'row_min', 'row_mean', 'row_var']].groupby(['sequence'])
    res = Parallel(n_jobs=32)(delayed(get_row_t)(name, g) for name, g in tqdm(gp))
    df_row = pd.concat(res)
    return df_row


def get_col_diff_stats(df, features, diff_features):
    data = df.drop('step', axis=1)
    del df
    gc.collect()
    gp = data.groupby(['sequence'], as_index=False)[diff_features]

    gp_max = gp.max()
    # gp_max.columns = ['sequence'] + [f'sensor_{i}_diff_max' for i in range(10)]
    gp_max.columns = ['sequence'] + \
        [f'{col}_lag1_max' for col in features] + \
            [f'{col}_lag2_max' for col in features] + \
                [f'{col}_diff1_max' for col in features] + \
                    [f'{col}_diff2_max' for col in features]

    gp_min = gp.min()
    # gp_min.columns = ['sequence'] + [f'sensor_{i}_diff_min' for i in range(10)]
    gp_min.columns = ['sequence'] + \
        [f'{col}_lag1_min' for col in features] + \
            [f'{col}_lag2_min' for col in features] + \
                [f'{col}_diff1_min' for col in features] + \
                    [f'{col}_diff2_min' for col in features]

    gp_mean = gp.mean()
    # gp_sum.columns = ['sequence'] + [f'sensor_{i}_diff_mean' for i in range(10)]
    gp_mean.columns = ['sequence'] + \
        [f'{col}_lag1_mean' for col in features] + \
            [f'{col}_lag2_mean' for col in features] + \
                [f'{col}_diff1_mean' for col in features] + \
                    [f'{col}_diff2_mean' for col in features]

    gp_var = gp.var()
    # gp_var.columns = ['sequence'] + [f'sensor_{i}_diff_var' for i in range(10)]
    gp_var.columns = ['sequence'] + \
        [f'{col}_lag1_var' for col in features] +\
            [f'{col}_lag2_var' for col in features] + \
                [f'{col}_diff1_var' for col in features] + \
                    [f'{col}_diff2_var' for col in features]

    df = gp_max.merge(gp_min, on=['sequence'], how='left')
    df = df.merge(gp_mean, on=['sequence'], how='left')
    df = df.merge(gp_var, on=['sequence'], how='left')
    return df


def get_col_diff_stats_parallel(df, features, diff_features):
    sequences = df['sequence'].unique().tolist()
    sequences_gp = []
    for i in range(0, len(sequences), 500):
        sequences_gp.append(df[df['sequence'].isin(sequences[i: i+500])])
    res = Parallel(backend='multiprocessing', n_jobs=32)(delayed(get_col_diff_stats)(df_tmp, features, diff_features) for df_tmp in tqdm(sequences_gp))
    df = pd.concat(res, ignore_index=True)
    return df


def main():
    DEBUG = False
    path = './output/'
    to_path = './output/'
    if not os.path.exists(to_path):
        os.makedirs(to_path)

    train, train_b, test, train_labels, train_labels_b = load_data(path)

    if DEBUG:
        train = train.head(672 * 500)
        train_b = train_b.head(672 * 500)
        test = test.head(672 * 500)
        train_labels = train_labels.head(500)
        train_labels_b = train_labels_b.head(500)

    features = ['sensor_{}'.format(i) for i in range(10)]
    
    train = get_feat(train, features)
    print(f'train: {train.shape}')

    train_b = get_feat(train_b, features)
    print(f'train_b: {train_b.shape}')

    test = get_feat(test, features)
    print(f'test: {test.shape}')

    diff_features = [f'{col}_lag1' for col in features] + [f'{col}_lag2' for col in features] + [f'{col}_diff1' for col in features] + [f'{col}_diff2' for col in features]

    # col diff stats
    train_col_diff_stats = get_col_diff_stats_parallel(train, features, diff_features)
    print(f'train_col_diff_stats: {train_col_diff_stats.shape}')
    train_col_diff_stats.reset_index(drop=True, inplace=True)
    train_col_diff_stats.to_feather(to_path + 'train_col_diff_stats.feather')
    del train_col_diff_stats
    gc.collect()

    train_b_col_diff_stats = get_col_diff_stats_parallel(train_b, features, diff_features)
    print(f'train_b_col_diff_stats: {train_b_col_diff_stats.shape}')
    train_b_col_diff_stats.reset_index(drop=True, inplace=True)
    train_b_col_diff_stats.to_feather(to_path + 'train_b_col_diff_stats.feather')
    del train_b_col_diff_stats
    gc.collect()

    test_col_diff_stats = get_col_diff_stats_parallel(test, features, diff_features)
    print(f'test_col_diff_stats: {test_col_diff_stats.shape}')
    test_col_diff_stats.reset_index(drop=True, inplace=True)
    test_col_diff_stats.to_feather(to_path + 'test_col_diff_stats.feather')
    del test_col_diff_stats
    gc.collect()
    
    # col stats
    train_col_stats = get_col_stats_parallel(train, features)
    print(f'train cols stats: {train_col_stats.shape}')
    train_col_stats.reset_index(drop=True, inplace=True)
    train_col_stats.to_feather(to_path + 'train_col_stats.feather')
    del train_col_stats
    gc.collect()

    train_b_col_stats = get_col_stats_parallel(train_b, features)
    print(f'train_b cols stats: {train_b_col_stats.shape}')
    train_b_col_stats.reset_index(drop=True, inplace=True)
    train_b_col_stats.to_feather(to_path + 'train_b_col_stats.feather')
    del train_b_col_stats
    gc.collect()

    test_col_stats = get_col_stats_parallel(test, features)
    print(f'test cols stats: {test_col_stats.shape}')
    test_col_stats.reset_index(drop=True, inplace=True)
    test_col_stats.to_feather(to_path + 'test_col_stats.feather')
    del test_col_stats
    gc.collect()

    # sensor t
    train_sensor_t = get_sensor_t_parallel(train)
    print(f'train sensor t: {train_sensor_t.shape}')
    train_sensor_t.reset_index(drop=True, inplace=True)
    train_sensor_t.to_feather(to_path + 'train_sensor_t.feather')
    del train_sensor_t
    gc.collect()

    train_b_sensor_t = get_sensor_t_parallel(train_b)
    print(f'train_b sensor t: {train_b_sensor_t.shape}')
    train_b_sensor_t.reset_index(drop=True, inplace=True)
    train_b_sensor_t.to_feather(to_path + 'train_b_sensor_t.feather')
    del train_b_sensor_t
    gc.collect()

    test_sensor_t = get_sensor_t_parallel(test)
    print(f'test sensor t: {test_sensor_t.shape}')
    test_sensor_t.reset_index(drop=True, inplace=True)
    test_sensor_t.to_feather(to_path + 'test_sensor_t.feather')
    del test_sensor_t
    gc.collect()

    train_row_t = get_row_t_parallel(train)
    print(f'train row t: {train_row_t.shape}')
    train_row_t.reset_index(drop=True, inplace=True)
    train_row_t.to_feather(to_path + 'train_row_t.feather')
    del train_row_t
    gc.collect()

    train_b_row_t = get_row_t_parallel(train_b)
    print(f'train_b row t: {train_b_row_t.shape}')
    train_b_row_t.reset_index(drop=True, inplace=True)
    train_b_row_t.to_feather(to_path + 'train_b_row_t.feather')
    del train_b_row_t
    gc.collect()

    test_row_t = get_row_t_parallel(test)
    print(f'test row t: {test_row_t.shape}')
    test_row_t.reset_index(drop=True, inplace=True)
    test_row_t.to_feather(to_path + 'test_row_t.feather')
    del test_row_t
    gc.collect()


if __name__ == '__main__':
    main()

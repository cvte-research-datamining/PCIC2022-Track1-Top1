import numpy as np
import pandas as pd
import gc
import os
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics import f1_score
from catboost import CatBoostClassifier, Pool
import warnings
warnings.filterwarnings('ignore')


def load_data():
    path = '../../01-feat/output/'
    # path2 = '../../input/dataset3_sensor_process/'
    train_col_stats = pd.read_feather(path + 'train_col_stats.feather')
    train_sensor_t = pd.read_feather(path + 'train_sensor_t.feather')
    train_row_t = pd.read_feather(path + 'train_row_t.feather')
    train_col_diff_stats = pd.read_feather(path + 'train_col_diff_stats.feather')
    train_labels = pd.read_feather(path + 'train_labels.feather')

    train = train_col_stats.merge(train_sensor_t, on='sequence', how='left')
    train = train.merge(train_row_t, on='sequence', how='left')
    train = train.merge(train_col_diff_stats, on='sequence', how='left')
    train = train.merge(train_labels, on='sequence', how='left')
    del train_col_stats, train_sensor_t, train_row_t, train_col_diff_stats, train_labels
    gc.collect()

    train_b_col_stats = pd.read_feather(path + 'train_b_col_stats.feather')
    train_b_sensor_t = pd.read_feather(path + 'train_b_sensor_t.feather')
    train_b_row_t = pd.read_feather(path + 'train_b_row_t.feather')
    train_b_col_diff_stats = pd.read_feather(path + 'train_b_col_diff_stats.feather')
    train_labels_b = pd.read_feather(path + 'train_labels_b.feather')
    
    train_b = train_b_col_stats.merge(train_b_sensor_t, on='sequence', how='left')
    train_b = train_b.merge(train_b_row_t, on='sequence', how='left')
    train_b = train_b.merge(train_b_col_diff_stats, on='sequence', how='left')
    train_b = train_b.merge(train_labels_b, on='sequence', how='left')
    del train_b_col_stats, train_b_sensor_t, train_b_row_t, train_b_col_diff_stats, train_labels_b
    gc.collect()

    test_col_stats = pd.read_feather(path + 'test_col_stats.feather')
    test_sensor_t = pd.read_feather(path + 'test_sensor_t.feather')
    test_row_t = pd.read_feather(path + 'test_row_t.feather')
    test_col_diff_stats = pd.read_feather(path + 'test_col_diff_stats.feather')

    test = test_col_stats.merge(test_sensor_t, on='sequence', how='left')
    test = test.merge(test_row_t, on='sequence', how='left')
    test = test.merge(test_col_diff_stats, on='sequence', how='left')
    del test_col_stats, test_sensor_t, test_row_t, test_col_diff_stats
    gc.collect()
    return train, train_b, test


if __name__ == '__main__':
    DEBUG = False
    n_splits = 10
    seed = 537821

    print('load data...')
    train, train_b, test = load_data()
    train = pd.concat([train, train_b], ignore_index=True)
    del train_b
    gc.collect()

    if DEBUG:
        train = train.head(1000)
        test = test.head(1000)
        train.reset_index(drop=True, inplace=True)
        test.reset_index(drop=True, inplace=True)
    
    print('train shape: ', train.shape)
    print('test shape: ', test.shape)
    
    sub_test = test[['sequence']]
    sub_test.reset_index(drop=True, inplace=True)
    sub_test['prob'] = 0

    params = {
        'iterations': 100000,
        'learning_rate': 0.1,
        'depth': 10,
        'l2_leaf_reg': 3,
        'loss_function': 'Logloss',
        'eval_metric': 'F1',
        'task_type': 'GPU',
        'devices': '0',
        'random_seed': 2021
    }
    
    cols = [col for col in train.columns if col not in ['sequence', 'label']]
    y_train = train['label']
    sub_valids = []
    # folds = StratifiedKFold(n_splits=10, shuffle=True, random_state=2021)
    folds = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for fold_, (trn_idx, val_idx) in enumerate(folds.split(train, y_train)):
        print("fold {}".format(fold_ + 1))

        X_tra, X_val = train.iloc[trn_idx][cols], train.iloc[val_idx][cols]
        y_tra, y_val = y_train.iloc[trn_idx], y_train.iloc[val_idx]
        print('train shape : ', X_tra.shape, y_tra.shape)
        print('valid shape : ', X_val.shape, y_val.shape)
        valid_label1_count = np.sum(y_val == 1)
        print('valid label 1 count: ', valid_label1_count)

        sub_valid = train.iloc[val_idx][['sequence', 'label']]
        sub_valid.reset_index(drop=True, inplace=True)
        sub_valid.rename(columns={'label': 'true'}, inplace=True)

        tra_data = Pool(X_tra, y_tra)
        val_data = Pool(X_val, y_val)

        print('training...')
        model = CatBoostClassifier(**params)
        model.fit(
            tra_data,
            eval_set=val_data,
            early_stopping_rounds=200,
            verbose=500
        )
        print('train done!')
        
        sub_valid['prob'] = model.predict_proba(X_val)[:, 1]
        sub_valid['rank'] = sub_valid['prob'].rank()
        sub_valid['pred'] = np.where(sub_valid['rank'] > (len(sub_valid) - valid_label1_count), 1, 0)
        print('valid f1 score: ', f1_score(sub_valid['true'], sub_valid['pred']))
        sub_valids.append(sub_valid)

        sub_test['prob'] += model.predict_proba(test.drop(['sequence'], axis=1))[:, 1] / folds.n_splits
    sub_test.to_csv('prob.csv', index=False)
    
    df_valid = pd.concat(sub_valids, ignore_index=True)
    print('\nvalid f1 score: ', f1_score(df_valid['true'], df_valid['pred']))
    df_valid.to_csv('valid.csv', index=False)
    print('All done!')

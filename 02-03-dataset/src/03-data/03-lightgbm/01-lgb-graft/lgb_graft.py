import numpy as np
import pandas as pd
import gc
from sklearn.model_selection import train_test_split, StratifiedKFold, KFold
from sklearn.metrics import f1_score
import lightgbm as lgb
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


def lgb_f1_score(y_hat, data):
    y_true = data.get_label()
    df = pd.DataFrame({'true': y_true, 'prob': y_hat})
    df['rank'] = df['prob'].rank()
    df['pred'] = np.where(df['rank'] > (len(df) - valid_label1_count), 1, 0)
    return 'f1', f1_score(df['true'], df['pred']), True


if __name__ == '__main__':
    DEBUG = False
    n_splits = 10
    seed1 = 661188
    seed2 = 6659

    print('load data...')
    train, train_b, test = load_data()

    if DEBUG:
        n_splits = 2
        train = train.head(1000)
        test = test.head(1000)
    
    train['city'] = 0
    train_b['city'] = 1
    test['city'] = 1

    train = pd.concat([train, train_b], ignore_index=True)
    del train_b
    gc.collect()
    
    print('train shape: ', train.shape)
    print('test shape: ', test.shape)
    
    sub_test1 = test[['sequence']]
    sub_test1.reset_index(drop=True, inplace=True)
    sub_test1['prob'] = 0

    sub_test2 = test[['sequence']]
    sub_test2.reset_index(drop=True, inplace=True)
    sub_test2['prob'] = 0

    params = {
        'objective': 'binary',
        'boosting': 'gbdt',
        # 'metric': 'auc',
        'metric': 'None',  # 用自定义评估函数是将metric设置为'None'
        'learning_rate': 0.1,
        'num_leaves': 31,
        'lambda_l1': 0,
        'lambda_l2': 1,
        'num_threads': 48,
        'min_data_in_leaf': 20,
        'first_metric_only': True,
        'is_unbalance': True,
        'max_depth': -1,
        'verbose': -1,
        'seed': 2020
    }
    
    cols = [col for col in train.columns if col not in ['sequence', 'label']]
    y_train = train['label']
    sub_valids = []
    # folds = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=2021)
    folds = KFold(n_splits=n_splits, shuffle=True, random_state=seed1)
    for fold_, (trn_idx, val_idx) in enumerate(folds.split(train, y_train)):
        X_tra, X_val = train.iloc[trn_idx][cols], train.iloc[val_idx][cols]
        y_tra, y_val = y_train.iloc[trn_idx], y_train.iloc[val_idx]
        valid_label1_count = np.sum(y_val == 1)
        print(f'fold {fold_ + 1}, valid label 1 count: {valid_label1_count}')

        sub_valid = train.iloc[val_idx][['sequence', 'label']]
        sub_valid.rename(columns={'label': 'true'}, inplace=True)
        sub_valid.reset_index(drop=True, inplace=True)

        dtrain = lgb.Dataset(X_tra, y_tra)
        dvalid = lgb.Dataset(X_val, y_val, reference=dtrain)

        model = lgb.train(
            params,
            dtrain,
            valid_sets=[dvalid, dtrain],
            num_boost_round=10000,
            early_stopping_rounds=200,
            verbose_eval=200,
            feval=lgb_f1_score
        )
        model.save_model(f'lgb_phase1_{fold_}.txt')
        print('Train done!')

        sub_valid['prob'] = model.predict(X_val)
        sub_valid['rank'] = sub_valid['prob'].rank()
        sub_valid['pred'] = np.where(sub_valid['rank'] > (len(sub_valid) - valid_label1_count), 1, 0)
        print('valid f1 score: ', f1_score(sub_valid['true'], sub_valid['pred']))
        sub_valids.append(sub_valid)

        sub_test1['prob'] += model.predict(test.drop(['sequence'], axis=1)) / folds.n_splits
    sub_test1.to_csv('prob.csv', index=False)
    
    df_valid = pd.concat(sub_valids, ignore_index=True)
    print('\nall valid f1 score: ', f1_score(df_valid['true'], df_valid['pred']))
    df_valid.to_csv('valid.csv', index=False)
    print('Phase 1 done!')

    # phase 2
    df_valid = pd.read_csv('valid.csv')
    test_prob = pd.read_csv('prob.csv')
    
    train = train.merge(df_valid[['sequence', 'prob']], on='sequence', how='left')
    # train = train[train['city'] == 1]
    gc.collect()
    print('train.shape: ', train.shape)

    cols = [col for col in train.columns if col not in ['sequence', 'label', 'city']]
    print('feats length', len(cols))
    y_train = train['label']
    sub_valids = []
    folds = KFold(n_splits=n_splits, shuffle=True, random_state=seed2)
    for fold_, (trn_idx, val_idx) in enumerate(folds.split(train, y_train)):
        print("fold {}".format(fold_ + 1))
        X_tra, X_val = train.iloc[trn_idx][cols], train.iloc[val_idx][cols]
        y_tra, y_val = y_train.iloc[trn_idx], y_train.iloc[val_idx]
        valid_label1_count = np.sum(y_val == 1)
        print('valid label 1 count: ', valid_label1_count)

        sub_valid = train.iloc[val_idx][['sequence', 'label']]
        sub_valid.rename(columns={'label': 'true'}, inplace=True)
        sub_valid.reset_index(drop=True, inplace=True)

        dtrain = lgb.Dataset(X_tra, y_tra)
        dvalid = lgb.Dataset(X_val, y_val, reference=dtrain)

        model = lgb.train(
            params,
            dtrain,
            valid_sets=[dvalid, dtrain],
            num_boost_round=10000,
            early_stopping_rounds=200,
            verbose_eval=200,
            feval=lgb_f1_score
        )
        model.save_model(f'lgb_phase2_{fold_+1}.txt')
        print('Train done!')

        sub_valid['prob'] = model.predict(X_val)
        sub_valid['rank'] = sub_valid['prob'].rank()
        sub_valid['pred'] = np.where(sub_valid['rank'] > (len(sub_valid) - valid_label1_count), 1, 0)
        print('valid f1 score: ', f1_score(sub_valid['true'], sub_valid['pred']))
        sub_valids.append(sub_valid)

        sub_test2['prob'] += model.predict(test.drop(['sequence'], axis=1)) / folds.n_splits
    sub_test2.to_csv('prob2.csv', index=False)
    
    df_valid = pd.concat(sub_valids, ignore_index=True)
    print('\nall valid f1 score: ', f1_score(df_valid['true'], df_valid['pred']))
    df_valid.to_csv('valid2.csv', index=False)
    print('Phase 2 done!')

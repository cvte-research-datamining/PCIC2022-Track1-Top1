import numpy as np
import pandas as pd
import os


def main():
    path = '../../../../dataset/'
    to_path = '../input/'
    if not os.path.exists(to_path):
        os.makedirs(to_path)

    # city A
    X_a = np.load(path + 'cityA/X.npy')
    y_a = np.load(path + 'cityA/Y.npy')
    print('city a origin shape: ', X_a.shape, y_a.shape)

    a_list = []
    for i in range(len(X_a)):
        df_tmp = pd.DataFrame(X_a[i])
        df_tmp.columns = [f'sensor_{i}' for i in range(10)]
        df_tmp['sequence'] = i
        df_tmp['step'] = range(672)
        a_list.append(df_tmp)
    df_a = pd.concat(a_list, ignore_index=True)
    df_a = df_a[['sequence', 'step'] + [f'sensor_{i}' for i in range(10)]]
    df_a.to_feather(to_path + 'df_a.feather')

    df_a_y = pd.DataFrame(y_a)
    df_a_y.columns = ['label']
    df_a_y['sequence'] = range(len(y_a))
    df_a_y = df_a_y[['sequence', 'label']]
    df_a_y.to_feather(to_path + 'df_a_y.feather')
    print('city a df shape: ', df_a.shape, df_a_y.shape)

    # city B
    X_b = np.load(path + 'cityB/train/X.npy')
    y_b = np.load(path + 'cityB/train/Y.npy')
    print('city b origin shape: ', X_b.shape, y_b.shape)

    b_list = []
    for i in range(len(X_b)):
        df_tmp = pd.DataFrame(X_b[i])
        df_tmp.columns = [f'sensor_{i}' for i in range(10)]
        df_tmp['sequence'] = i
        df_tmp['step'] = range(672)
        b_list.append(df_tmp)
    df_b = pd.concat(b_list, ignore_index=True)
    df_b = df_b[['sequence', 'step'] + [f'sensor_{i}' for i in range(10)]]
    df_b.to_feather(to_path + 'df_b.feather')

    df_b_y = pd.DataFrame(y_b)
    df_b_y.columns = ['label']
    df_b_y['sequence'] = range(len(y_b))
    df_b_y = df_b_y[['sequence', 'label']]
    df_b_y.to_feather(to_path + 'df_b_y.feather')
    print('city b df shape: ', df_b.shape, df_b_y.shape)

    X = np.load(path + 'cityB/test/X.npy')
    print('test origin shape: ', X.shape)

    b_test_list = []
    for i in range(len(X)):
        df_tmp = pd.DataFrame(X[i])
        df_tmp.columns = [f'sensor_{i}' for i in range(10)]
        df_tmp['sequence'] = i
        df_tmp['step'] = range(672)
        b_test_list.append(df_tmp)
    df_b_test = pd.concat(b_test_list, ignore_index=True)
    df_b_test = df_b_test[['sequence', 'step'] + [f'sensor_{i}' for i in range(10)]]
    df_b_test.to_feather(to_path + 'df_b_test.feather')
    print('city b df test shape: ', df_b_test.shape)


if __name__ == '__main__':
    main()

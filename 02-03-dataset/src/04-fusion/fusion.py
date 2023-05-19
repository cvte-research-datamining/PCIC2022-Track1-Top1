import numpy as np
import pandas as pd
import os


def main():
    path = '../../../../results/CVTEDMer_2/'
    if not os.path.exists(path):
        os.makedirs(path)

    path1 = '../02-data/finetune/output/'
    path2 = '../02-data/lstm/output/'
    path3 = '../03-data/02-catboost/kfold/' # 0.925875252
    path4= '../03-data/02-catboost/graft/' # 0.926356495 0.933986304
    path5 = '../03-data/03-lightgbm/01-lgb-graft/' # 0.922994649, 0.927279361

    prob1 = pd.read_csv(path1 + 'prob.csv')
    prob2 = pd.read_csv(path2 + 'prob.csv')

    prob3 = pd.read_csv(path3 + 'prob.csv')   # 0.925875252
    prob4 = pd.read_csv(path4 + 'prob.csv')   # 0.926356495
    prob5 = pd.read_csv(path4 + 'prob2.csv')  # 0.933986304
    prob6 = pd.read_csv(path5 + 'prob.csv')   # 0.922994649
    prob7 = pd.read_csv(path5 + 'prob2.csv')  # 0.927279361

    prob1['prob'] = 0.51 * prob1['prob_mean'] + 0.49 * prob2['prob_mean']
    prob1['rank'] = prob1['prob'].rank()

    prob3['prob'] = 0.17 * prob3['prob'] + 0.15 * prob4['prob'] + 0.33 * prob5['prob'] + 0.1 * prob6['prob'] + 0.25 * prob7['prob']
    prob3['rank'] = prob3['prob'].rank()

    prob1['label'] = np.where(prob1['rank'] > (len(prob1) - 6460), 1, 0)
    prob3['label'] = np.where(prob3['rank'] > (len(prob3) - 3280), 1, 0)

    prob = pd.concat([prob1, prob3], ignore_index=True)
    # prob['label'] = prob['label'].map({0: 1, 1: 0})
    prob[['label']].to_csv(path + 'submission.csv', index=False, header=False)

    print(prob[['label']].info())
    print(prob['label'].value_counts())


if __name__ == '__main__':
    main()
